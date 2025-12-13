# === AUDIO CHUNKING - dla długich plików ===

def split_audio_chunks(audio_path: str, chunk_duration_minutes: int = 5) -> list:
    """
    Dzieli audio na chunki używając pydub.
    Zwraca listę ścieżek do chunków.
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_duration_ms = chunk_duration_minutes * 60 * 1000

        chunks = []
        chunk_paths = []

        for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            chunk = audio[start_ms:end_ms]

            # Save chunk
            chunk_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            chunk.export(chunk_path, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
            chunk_paths.append({
                "path": chunk_path,
                "index": i,
                "start_time": start_ms / 1000,
                "end_time": end_ms / 1000,
                "duration": (end_ms - start_ms) / 1000
            })

        logger.info(f"Audio podzielone na {len(chunk_paths)} chunków po {chunk_duration_minutes} min")
        return chunk_paths

    except Exception as e:
        logger.error(f"Błąd dzielenia audio: {e}")
        return []


def extract_audio_chunked(file, chunk_duration_minutes: int = 5, generate_summary: bool = True):
    """
    Audio → tekst przez Whisper z podziałem na chunki.

    1. Dzieli długie audio na chunki (np. 5 min)
    2. Przetwarza każdy chunk osobno przez Whisper
    3. Łączy transkrypcje
    4. Generuje AI summary (opcjonalne)

    Zwraca (text, pages, meta)
    """
    import streamlit as st

    try:
        size_bytes = get_file_size(file)
        file.seek(0)
        raw = file.read()
        fname = file.name

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(fname)[1], delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # Sprawdź długość audio
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_path)
            duration_seconds = len(audio) / 1000
            duration_minutes = duration_seconds / 60

            logger.info(f"Audio długość: {duration_minutes:.1f} minut")

            # Jeśli krótkie audio (< chunk_duration), użyj standardowej metody
            if duration_minutes < chunk_duration_minutes:
                logger.info("Audio krótkie - przetwarzam bez chunków")
                os.unlink(tmp_path)
                file.seek(0)
                return extract_audio_whisper(file)

        except Exception as e:
            logger.warning(f"Nie można określić długości audio: {e}, proceduję z chunkowaniem")

        # Podziel na chunki
        st.info(f"🎵 Dzielę audio na {chunk_duration_minutes}-minutowe odcinki...")
        chunks = split_audio_chunks(tmp_path, chunk_duration_minutes)

        if not chunks:
            # Fallback do standardowej metody
            os.unlink(tmp_path)
            file.seek(0)
            return extract_audio_whisper(file)

        # Przetwarzaj każdy chunk
        all_segments = []
        all_texts = []
        total_chunks = len(chunks)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, chunk_info in enumerate(chunks):
            chunk_path = chunk_info["path"]
            chunk_idx = chunk_info["index"]
            chunk_start = chunk_info["start_time"]

            status_text.text(f"🎙️ Przetwarzam chunk {i+1}/{total_chunks} (rozpoczęcie: {chunk_start/60:.1f} min)...")

            try:
                # Przeczytaj chunk
                with open(chunk_path, 'rb') as f:
                    chunk_raw = f.read()

                # Wyślij do Whisper z krótszym timeout (chunk jest mały)
                files = {"audio_file": (f"chunk_{chunk_idx}.wav", chunk_raw, "audio/wav")}
                timeout_read = 180  # 3 minuty wystarczy dla 5-min chunka

                r = http_post(
                    f"{WHISPER_URL}/asr?task=transcribe&language=pl&word_timestamps=true&output=json",
                    files=files,
                    timeout=(30, timeout_read)
                )
                r.raise_for_status()

                result = r.json()
                chunk_text = result.get("text", "")
                chunk_segments = result.get("segments", [])

                # Przesunięcie timestampów o offset chunka
                for seg in chunk_segments:
                    seg["start"] += chunk_start
                    seg["end"] += chunk_start
                    seg["chunk_index"] = chunk_idx

                all_segments.extend(chunk_segments)
                all_texts.append(f"[Chunk {chunk_idx+1}, {chunk_start/60:.1f}-{chunk_info['end_time']/60:.1f} min]\n{chunk_text}")

                logger.info(f"Chunk {chunk_idx} przetworzony: {len(chunk_text)} znaków")

            except Exception as e:
                logger.error(f"Błąd przetwarzania chunk {chunk_idx}: {e}")
                all_texts.append(f"[Chunk {chunk_idx+1}] BŁĄD: {e}")

            finally:
                # Cleanup chunk file
                try:
                    os.unlink(chunk_path)
                except Exception:
                    pass

            # Update progress
            progress_bar.progress((i + 1) / total_chunks)

        progress_bar.empty()
        status_text.empty()

        # Cleanup original temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # Połącz transkrypcje
        full_text = "\n\n".join(all_texts)

        # Opcjonalne: generuj AI summary
        summary_text = ""
        if generate_summary and all_texts:
            st.info("🤖 Generuję podsumowanie AI...")
            try:
                # Przygotuj prompt
                combined_transcript = "\n".join([t.split("\n", 1)[1] if "\n" in t else t for t in all_texts])

                summary_prompt = f"""Przeanalizuj poniższą transkrypcję audio i stwórz zwięzłe podsumowanie.

TRANSKRYPCJA:
{combined_transcript[:8000]}

ZADANIE:
1. Główne tematy (3-5 punktów)
2. Kluczowe informacje
3. Wnioski (jeśli są)

Format: krótkie, konkretne punkty.
"""

                summary = query_ollama_text(summary_prompt, timeout=60)
                if summary and not summary.startswith("[BŁĄD"):
                    summary_text = f"\n\n{'='*60}\n=== AI PODSUMOWANIE ===\n{'='*60}\n\n{summary}\n"

            except Exception as e:
                logger.error(f"Błąd generowania podsumowania: {e}")

        # Finalny tekst z timestampami
        lines = ["=== TRANSKRYPCJA Z CHUNKÓW (z timestampami) ===", ""]
        for seg in all_segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            txt = (seg.get("text") or "").strip()
            chunk_idx = seg.get("chunk_index", 0)
            lines.append(f"[{start/60:.1f}min - {end/60:.1f}min | Chunk {chunk_idx+1}] {txt}")

        full_text_with_timestamps = "\n".join(lines)

        # Dodaj summary na końcu
        if summary_text:
            full_text_with_timestamps += summary_text

        meta = {
            "type": "audio",
            "chunks": total_chunks,
            "chunk_duration_minutes": chunk_duration_minutes,
            "total_segments": len(all_segments),
            "duration": all_segments[-1]["end"] if all_segments else 0,
            "has_summary": bool(summary_text),
            "segments": all_segments
        }

        return full_text_with_timestamps, 1, meta

    except Exception as e:
        logger.error(f"Chunked audio error: {e}")
        return f"[BŁĄD CHUNKED AUDIO: {e}]", 0, {"type": "audio", "error": str(e)}
