import logging
import os
import queue
import time
from datetime import datetime, timedelta

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

# Frames quieter than this (normalized RMS, i.e. rms/32768) are treated as
# silence and never reach a transcription provider.
#
# WHY THIS IS CONFIGURABLE: the original hardcoded 0.01 is ~-40 dBFS, which is
# far too aggressive for the per-participant audio Google Meet hands a bot. On
# real interviews it discarded ~95% of frames, and a quieter participant could
# be dropped ENTIRELY — process_chunk() only opens a buffer for a speaker on
# their first non-silent frame, so a speaker who never clears this bar produces
# no utterances at all and vanishes from the transcript. That happened to the
# candidate in three interviews before it was caught.
#
# webrtcvad still runs after this check, so this is only a cheap pre-filter;
# lowering it hands more frames to a purpose-built voice detector rather than
# letting an amplitude heuristic decide alone.
#
# Default preserves upstream behaviour when the env var is unset.
DEFAULT_SILENCE_RMS_THRESHOLD = 0.01


def _silence_rms_threshold():
    raw = os.getenv("AUDIO_SILENCE_RMS_THRESHOLD")
    if not raw:
        return DEFAULT_SILENCE_RMS_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"AUDIO_SILENCE_RMS_THRESHOLD is not a number: {raw!r}; using {DEFAULT_SILENCE_RMS_THRESHOLD}")
        return DEFAULT_SILENCE_RMS_THRESHOLD
    if not 0 <= value < 1:
        logger.warning(f"AUDIO_SILENCE_RMS_THRESHOLD out of range (0 <= x < 1): {value}; using {DEFAULT_SILENCE_RMS_THRESHOLD}")
        return DEFAULT_SILENCE_RMS_THRESHOLD
    return value


def calculate_normalized_rms(audio_bytes):
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(samples)))
    # Normalize by max possible value for 16-bit audio (32768)
    return rms / 32768


class PerParticipantNonStreamingAudioInputManager:
    def __init__(self, *, save_audio_chunk_callback, get_participant_callback, sample_rate, utterance_size_limit, silence_duration_limit, should_print_diagnostic_info):
        self.queue = queue.Queue()

        self.save_audio_chunk_callback = save_audio_chunk_callback
        self.get_participant_callback = get_participant_callback

        self.utterances = {}
        self.sample_rate = sample_rate

        self.first_nonsilent_audio_time = {}
        self.last_nonsilent_audio_time = {}

        self.UTTERANCE_SIZE_LIMIT = utterance_size_limit
        self.SILENCE_DURATION_LIMIT = silence_duration_limit
        self.SILENCE_RMS_THRESHOLD = _silence_rms_threshold()
        self.vad = webrtcvad.Vad()
        logger.info(f"PerParticipantNonStreamingAudioInputManager silence RMS threshold: {self.SILENCE_RMS_THRESHOLD}")

        self.should_print_diagnostic_info = should_print_diagnostic_info
        self.reset_diagnostic_info()

    def add_chunk(self, speaker_id, chunk_time, chunk_bytes):
        self.queue.put((speaker_id, chunk_time, chunk_bytes))
        self.diagnostic_info["total_chunks_added"] += 1

    def reset_diagnostic_info(self):
        self.diagnostic_info = {
            "total_chunks_added": 0,
            "total_chunks_marked_as_silent_due_to_vad": 0,
            "total_chunks_marked_as_silent_due_to_rms_being_small": 0,
            "total_chunks_marked_as_silent_due_to_rms_being_zero": 0,
            "total_chunks_too_large_for_vad": 0,
            "total_chunks_that_caused_vad_error": 0,
            "total_audio_chunks_sent": 0,
            "total_audio_chunks_not_sent_because_participant_not_found": 0,
            # RMS observability — without these, a too-high threshold is
            # indistinguishable from a genuinely silent room in the logs.
            "rms_threshold": self.SILENCE_RMS_THRESHOLD,
            "rms_max": 0.0,
            "rms_mean_of_nonzero": 0.0,
            # how many frames WOULD have passed at successively lower bars, so a
            # sensible threshold can be read straight off a real call
            "rms_above_threshold": 0,
            "rms_above_0_005": 0,
            "rms_above_0_002": 0,
            "rms_above_0_001": 0,
        }
        self._rms_sum = 0.0
        self._rms_count = 0
        self.last_diagnostic_info_print_time = time.time()

    def print_diagnostic_info(self):
        if time.time() - self.last_diagnostic_info_print_time >= 30:
            if self.should_print_diagnostic_info:
                logger.info(f"PerParticipantNonStreamingAudioInputManager diagnostic info: {self.diagnostic_info}")
            self.reset_diagnostic_info()

    def process_chunks(self):
        while not self.queue.empty():
            speaker_id, chunk_time, chunk_bytes = self.queue.get()
            self.process_chunk(speaker_id, chunk_time, chunk_bytes)

        for speaker_id in list(self.first_nonsilent_audio_time.keys()):
            self.process_chunk(speaker_id, datetime.utcnow(), None)

        self.print_diagnostic_info()

    # When the meeting ends, we need to flush all utterances. Do this by pretending that we received a chunk of silence at the end of the meeting.
    def flush_utterances(self):
        for speaker_id in list(self.first_nonsilent_audio_time.keys()):
            self.process_chunk(
                speaker_id,
                datetime.utcnow() + timedelta(seconds=self.SILENCE_DURATION_LIMIT + 1),
                None,
            )

    def is_speech(self, chunk_bytes):
        try:
            # The VAD can handle a max of 30 ms of audio. If it is larger than that, just return True
            if len(chunk_bytes) > 30 * self.sample_rate // 1000:
                self.diagnostic_info["total_chunks_too_large_for_vad"] += 1
                return True
            return self.vad.is_speech(chunk_bytes, self.sample_rate)
        except Exception as e:
            logger.exception("Error in VAD: " + str(e))
            self.diagnostic_info["total_chunks_that_caused_vad_error"] += 1
            return True

    def record_rms(self, rms_value):
        if rms_value > self.diagnostic_info["rms_max"]:
            self.diagnostic_info["rms_max"] = round(float(rms_value), 5)
        self._rms_sum += float(rms_value)
        self._rms_count += 1
        self.diagnostic_info["rms_mean_of_nonzero"] = round(self._rms_sum / self._rms_count, 5)
        if rms_value >= self.SILENCE_RMS_THRESHOLD:
            self.diagnostic_info["rms_above_threshold"] += 1
        for bar, key in ((0.005, "rms_above_0_005"), (0.002, "rms_above_0_002"), (0.001, "rms_above_0_001")):
            if rms_value >= bar:
                self.diagnostic_info[key] += 1

    def silence_detected(self, chunk_bytes):
        rms_value = calculate_normalized_rms(chunk_bytes)
        if rms_value == 0:
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_rms_being_zero"] += 1
            return True
        self.record_rms(rms_value)
        if rms_value < self.SILENCE_RMS_THRESHOLD:
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_rms_being_small"] += 1
            return True
        if not self.is_speech(chunk_bytes):
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_vad"] += 1
            return True
        return False

    def process_chunk(self, speaker_id, chunk_time, chunk_bytes):
        audio_is_silent = self.silence_detected(chunk_bytes) if chunk_bytes else True

        # Initialize buffer and timing for new speaker
        if speaker_id not in self.utterances or len(self.utterances[speaker_id]) == 0:
            if audio_is_silent:
                return
            self.utterances[speaker_id] = bytearray()
            self.first_nonsilent_audio_time[speaker_id] = chunk_time
            self.last_nonsilent_audio_time[speaker_id] = chunk_time

        # Add new audio data to buffer
        if chunk_bytes:
            self.utterances[speaker_id].extend(chunk_bytes)

        should_flush = False
        reason = None

        # Check buffer size
        if len(self.utterances[speaker_id]) >= self.UTTERANCE_SIZE_LIMIT:
            should_flush = True
            reason = "buffer_full"

        # Check for silence
        if audio_is_silent:
            silence_duration = (chunk_time - self.last_nonsilent_audio_time[speaker_id]).total_seconds()
            if silence_duration >= self.SILENCE_DURATION_LIMIT:
                should_flush = True
                reason = "silence_limit"
        else:
            self.last_nonsilent_audio_time[speaker_id] = chunk_time

            logger.debug(f"Speaker {speaker_id} is speaking")

        # Flush buffer if needed
        if should_flush and len(self.utterances[speaker_id]) > 0:
            participant = self.get_participant_callback(speaker_id)
            if participant:
                self.save_audio_chunk_callback(
                    {
                        **participant,
                        "audio_data": bytes(self.utterances[speaker_id]),
                        "timestamp_ms": int(self.first_nonsilent_audio_time[speaker_id].timestamp() * 1000),
                        "flush_reason": reason,
                        "sample_rate": self.sample_rate,
                    }
                )
                self.diagnostic_info["total_audio_chunks_sent"] += 1
            else:
                logger.warning(f"Participant {speaker_id} not found")
                self.diagnostic_info["total_audio_chunks_not_sent_because_participant_not_found"] += 1
            # Clear the buffer
            self.utterances[speaker_id] = bytearray()
            del self.first_nonsilent_audio_time[speaker_id]
            del self.last_nonsilent_audio_time[speaker_id]
