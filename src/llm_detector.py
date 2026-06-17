"""
Base LLM Detector for Hate Speech Detection (Ollama backend)
"""

import re
from pathlib import Path
import requests
from typing import List, Dict
from src.utils import parse_category_and_subcategory


def strip_markdown_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence from a model response."""
    value = str(text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", value, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else value


class LLMDetector:
    """Base class for hate speech detection using local Ollama service"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", default_temperature: float = 0.1, default_max_tokens: int = 1024, prompts_dir: Path = Path(__file__).parent / "prompts"):
        """
        Initialize the LLM detector (Ollama)

        Args:
            model_name: Ollama model tag (e.g., "llama3.2:3b", "phi3:mini", "mistral:7b")
            base_url: Base URL of the local Ollama server
            default_temperature: Default sampling temperature
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.max_tokens = default_max_tokens
        self._session = requests.Session()

        self.prompts_dir = prompts_dir
        prompt_path = self.prompts_dir / "system_prompt.txt"
        with open(prompt_path, encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    # --- Internal helpers ---
    def _clean_content(self, text: str) -> str:
        # Remove possible reasoning tags and trim
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Convert chat messages to a single prompt for /api/generate
        parts = []
        for m in messages:
            role = m.get("role", "user").strip()
            content = m.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _post(self, url: str, payload: Dict) -> requests.Response:
        return self._session.post(url, json=payload)

    def _chat(self, messages: List[Dict[str, str]], num_predict: int, temperature: float) -> str:
        # Try /api/chat first
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        chat_url = f"{self.base_url}/api/chat"
        resp = self._post(chat_url, chat_payload)
        if resp.status_code in (404, 405, 501):
            # Fallback to /api/generate (older Ollama versions)
            prompt = self._to_prompt(messages)
            gen_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            }
            gen_url = f"{self.base_url}/api/generate"
            gen_resp = self._post(gen_url, gen_payload)

            if gen_resp.status_code == 404 and "localhost" in self.base_url:
                # Try 127.0.0.1 fallback
                gen_url = gen_url.replace("localhost", "127.0.0.1")
                gen_resp = self._post(gen_url, gen_payload)

            gen_resp.raise_for_status()
            data = gen_resp.json()
            content = data.get("response", "")
            return self._clean_content(content)

        if resp.status_code == 404 and "localhost" in self.base_url:
            # Try 127.0.0.1 for /api/chat
            chat_url = chat_url.replace("localhost", "127.0.0.1")
            resp = self._post(chat_url, chat_payload)

        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("message") or {})
        content = (msg.get("content") or data.get("response") or "")

        return self._clean_content(content)

    # --- Public API (compatible with previous HF-based version) ---
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        temp = temperature if temperature is not None else self.default_temperature
        return self._chat(messages, num_predict=max_new_tokens, temperature=temp)

    def detect_hate_speech_binary(self, text: str) -> bool:
        """
        Zadatak 1: Utvrdi da li tekst sadrži govor mržnje (binarna klasifikacija)
        Vraća: (sadrži_govor_mržnje)
        """
        # Load prompt relative to this file to avoid CWD issues
        prompt_path = self.prompts_dir / "detect.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()

        prompt = prompt.format(text=text)

        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)
        contains_hate = False
        first_tokens = re.findall(r"\b\w+\b", response.lower())[:2]
        if any(tok in {"da"} for tok in first_tokens):
            contains_hate = True
        return contains_hate

    def categorize_hate_speech(self, text: str, categories_prompt: str) -> List[str]:
        """
        Zadatak 3: Klasifikuj govor mržnje u unapred definisane kategorije, dozvoljeno više.

        Vraća listu kodova poput ["4a", "6a"] ili ["2"] (ako nema podkategorije).
        """
        prompt_path = self.prompts_dir / "classify.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()  

        prompt = prompt.format(text=text, categories_prompt=categories_prompt)

        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)
        # Prvo probaj eksplicitnu liniju 'Kategorije: <lista>'
        codes: List[str] = []
        m_line = re.search(r"(?im)^\s*kategorije\s*:\s*(.+)$", response)
        if m_line:
            raw = m_line.group(1).strip()
            parts = [p.strip() for p in raw.split(",") if p is not None]
            for p in parts:
                p = p.replace(" ", "")
                if re.match(r"^[0-7][a-z]?$", p, flags=re.IGNORECASE):
                    codes.append(p.lower())

        # Ako linija nije data ili prazna, pokušaj regex-om svuda
        if not codes:
            # Ekstrakcija svih kodova kategorija iz odgovora
            # Podržava forme: "4a,6a", "Kategorije: 4a, 6a", "4, 6a" itd.
            codes = re.findall(r"\b([0-7][a-z]?)\b", response, flags=re.IGNORECASE)
            codes = [c.lower() for c in codes]

        # Ako ništa nije pronađeno, pokušaj da parsiraš polja Kategorija/Podkategorija
        if not codes:
            m_cat = re.search(r"(?i)kategorija\s*:\s*([0-7])\b", response)
            m_sub = re.search(r"(?i)podkategorija\s*:\s*([0-7][a-z])\b", response)
            if m_sub:
                codes = [m_sub.group(1).lower()]        
            elif m_cat:
                codes = [m_cat.group(1)]

        # Ukloni duplikate uz očuvanje redosleda
        seen = set()
        unique_codes: List[str] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes
    

    def categorize_hate_speech_few_shot(self, text: str, categories_prompt: str) -> List[str]:
        """
        Zadatak 3: Klasifikuj govor mržnje u unapred definisane kategorije, dozvoljeno više.

        Vraća listu kodova poput ["4a", "6a"] ili ["2"] (ako nema podkategorije).
        """
        prompt_path = self.prompts_dir / "classify_few_shot.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()  

        prompt = prompt.format(text=text, categories_prompt=categories_prompt)

        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)
        # Prvo probaj eksplicitnu liniju 'Kategorije: <lista>'
        codes: List[str] = []
        m_line = re.search(r"(?im)^\s*kategorije\s*:\s*(.+)$", response)
        if m_line:
            raw = m_line.group(1).strip()
            parts = [p.strip() for p in raw.split(",") if p is not None]
            for p in parts:
                p = p.replace(" ", "")
                if re.match(r"^[0-7][a-z]?$", p, flags=re.IGNORECASE):
                    codes.append(p.lower())

        # Ako linija nije data ili prazna, pokušaj regex-om svuda
        if not codes:
            # Ekstrakcija svih kodova kategorija iz odgovora
            # Podržava forme: "4a,6a", "Kategorije: 4a, 6a", "4, 6a" itd.
            codes = re.findall(r"\b([0-7][a-z]?)\b", response, flags=re.IGNORECASE)
            codes = [c.lower() for c in codes]

        # Ako ništa nije pronađeno, pokušaj da parsiraš polja Kategorija/Podkategorija
        if not codes:
            m_cat = re.search(r"(?i)kategorija\s*:\s*([0-7])\b", response)
            m_sub = re.search(r"(?i)podkategorija\s*:\s*([0-7][a-z])\b", response)
            if m_sub:
                codes = [m_sub.group(1).lower()]        
            elif m_cat:
                codes = [m_cat.group(1)]

        # Ukloni duplikate uz očuvanje redosleda
        seen = set()
        unique_codes: List[str] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes
    
    def detect_and_categorize(self, text: str, categories_prompt: str) -> Dict:
        """Jedan poziv koji detektuje govor mržnje i kategorizuje ga.

        Vraća dict:
        {
          'has_hate_speech': bool,
          'category': int (0-7),
          'subcategory': str ('' ili npr. '3b'),
          'raw': str (sirovi odgovor modela)
        }
        """
        prompt_path = self.prompts_dir / "detect_and_classify.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()

        prompt = prompt.format(text=text, categories_prompt=categories_prompt)
        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)

        # Parsiraj DA/NE
        has_hate = False
        m_hs = re.search(r"(?i)govormržnje\s*:\s*(da|ne)", response)
        if not m_hs:
            # fallback for potential ASCII variants
            m_hs = re.search(r"(?i)govor\s*mržnje\s*:\s*(da|ne)", response)
        if m_hs:
            has_hate = m_hs.group(1).strip().lower() == "da"

        # Kategorije (više vrednosti moguće) – prvo pokušaj eksplicitnu liniju 'Kategorije:'
        all_codes: List[str] = []
        m_line = re.search(r"(?im)^\s*kategorije\s*:\s*(.+)$", response)
        if m_line:
            raw_list = m_line.group(1).strip()
            # Ako model upiše samo '0' ili ostavi prazno
            if raw_list:
                parts = [p.strip() for p in raw_list.split(',') if p.strip()]
                for p in parts:
                    if re.match(r"^[0-7][a-z]?$", p, flags=re.IGNORECASE):
                        all_codes.append(p.lower())
            else:
                all_codes = []

        # Fallback: ako nije pronađena linija ili prazna, primeni regex globalno
        if not all_codes:
            regex_found = re.findall(r"\b([0-7][a-z]?)\b", response, flags=re.IGNORECASE)
            all_codes = [c.lower() for c in regex_found]
            # Ako ni to ne vrati ništa, pokušaj starim poljima 'Kategorija:' / 'Podkategorija:'
            if not all_codes:
                m_cat = re.search(r"(?i)kategorija\s*:\s*([0-7])\b", response)
                m_sub = re.search(r"(?i)podkategorija\s*:\s*([0-7][a-z])\b", response)
                if m_sub:
                    all_codes = [m_sub.group(1).lower()]
                elif m_cat:
                    all_codes = [m_cat.group(1)]

        # Normalizuj situacije: ako NE i nema kodova -> stavi ['0']; ako ima drugih kodova ukloni 0
        if not has_hate:
            if not all_codes or '0' not in all_codes:
                all_codes = ['0']
        else:
            if len(all_codes) > 1 and '0' in all_codes:
                all_codes = [c for c in all_codes if c != '0']

        # Očisti duplikate, očuvaj redosled
        seen = set()
        codes: List[str] = []
        for code in all_codes:
            if code not in seen:
                seen.add(code)
                codes.append(code)

        # Primarna kategorija za kompatibilnost (prva različita od 0 ako postoji)
        primary_cat = 0
        primary_sub = ""
        if codes:
            from src.utils import parse_category_and_subcategory as _parse
            for c in codes:
                parsed = _parse(c)
                if int(parsed.get("category", 0)) != 0:
                    primary_cat = int(parsed.get("category", 0))
                    primary_sub = str(parsed.get("subcategory", ""))
                    break
            if primary_cat == 0:
                # sve su nule ili ništa korisno
                parsed = _parse(codes[0])
                primary_cat = int(parsed.get("category", 0))
                primary_sub = str(parsed.get("subcategory", ""))

        return {
            "has_hate_speech": has_hate,
            "category": int(primary_cat),
            "subcategory": primary_sub,
            "codes": codes,
            "codes_str": ",".join(codes) if codes else "",
            "raw": response,
        }

    def classify_all_sentences(self, text: str, categories_prompt: str) -> Dict:
        """
        Podeli ceo tekst na rečenice i dodeli kategoriju svakoj.
        Koristi prompt 'classify_full_all.txt', format linije: (rečenica; kategorija; podkategorija)

        Vraća dict:
        {
          'sentences': List[{ 'sentence': str, 'category': int, 'subcategory': str }],
          'raw': str
        }
        """
        prompt_path = self.prompts_dir / "classify_full_all.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read()

        prompt = prompt.format(text=text, categories_prompt=categories_prompt)
        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)

        results: List[Dict[str, str]] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1].strip()
            parts = [p.strip().strip('"\'') for p in line.split(";")]
            if len(parts) < 2:
                continue
            sentence = parts[0]
            cat_part = (parts[1] if len(parts) >= 2 else "").lower()
            sub_part = (parts[2] if len(parts) >= 3 else "").lower()

            # cat_part could be '0' or '3' or '3b'. Prefer combined parse
            combined = cat_part if cat_part else sub_part
            parsed = parse_category_and_subcategory(combined)
            cat = int(parsed.get("category", 0))
            sub = parsed.get("subcategory", "")
            # If explicit sub given like '3b' or single letter, prefer it
            m_sub = re.match(r"^([0-7])\s*([a-z])$", sub_part)
            if m_sub:
                sub = m_sub.group(2)
            elif re.match(r"^[a-z]$", sub_part):
                sub = sub_part

            results.append({"sentence": sentence, "category": cat, "subcategory": sub})

        return {"sentences": results, "raw": response}

    def _pick_category_code(self, codes: list[str]) -> str:
        if not codes:
            raise ValueError("Expected at least one category code")
        unique_codes = []
        for code in codes:
            if code not in unique_codes:
                unique_codes.append(code)
        nonzero = [code for code in unique_codes if code != "0"]
        if not nonzero:
            return "0"
        categories = {code[0] for code in nonzero}
        if len(categories) == 1:
            specific = [code for code in nonzero if len(code) > 1]
            return specific[0] if specific else nonzero[0]
        return nonzero[0]

    def _category_code_from_plain_response(self, response: str, expected_id: int) -> str:
        cleaned = strip_markdown_code_fences(response).strip().lower()
        lines = [line.strip().lower() for line in cleaned.splitlines() if line.strip()]
        candidates = [cleaned] + lines
        for candidate in candidates:
            value = candidate.strip().strip("'\".,;:()[]{}")
            value = re.sub(r"\s+", "", value)
            if re.fullmatch(r"[0-7][a-z]?", value):
                return value
            if re.fullmatch(r"[0-7][a-z]?(?:[,;/]+[0-7][a-z]?)+", value):
                return self._pick_category_code(re.findall(r"[0-7][a-z]?", value))

        id_prefixed = re.search(
            rf"(?im)^\s*{expected_id}\s*[:=\-]\s*([0-7][a-z]?)\s*[\.;,]?\s*$",
            cleaned,
        )
        if id_prefixed:
            return id_prefixed.group(1).lower()

        labeled = re.search(
            r"(?im)^\s*(?:kategorija|category|odgovor|answer)\s*[:=\-]\s*([0-7][a-z]?)\s*[\.;,]?\s*$",
            cleaned,
        )
        if labeled:
            return labeled.group(1).lower()

        leading_code = re.search(r"^\s*([0-7][a-z]?)(?:\s*[,;]\s*|\s*\()", cleaned)
        if leading_code and len(cleaned) <= 120:
            return leading_code.group(1).lower()

        final_code = re.search(r"(?:^|[^0-9a-z])([0-7][a-z]?)\s*[\.;,]?\s*$", cleaned)
        if final_code and len(cleaned) <= 160:
            return final_code.group(1).lower()

        raise ValueError(f"Expected one category code, received: {cleaned[:120]}")

    def _normalize_sentence_code(self, code: str, expected_id: int) -> dict:
        code = str(code).strip().lower()
        if not re.fullmatch(r"[0-7][a-z]?", code):
            raise ValueError(f"Invalid category code: {code}")
        codes = ["0"] if code == "0" else [code]
        parsed = parse_category_and_subcategory(code)
        return {
            "id": expected_id,
            "has_hate_speech": code != "0",
            "category": int(parsed["category"]),
            "subcategory": str(parsed["subcategory"]),
            "codes": codes,
        }

    def classify_sentence_cot(
        self, sentence: str, sentence_id: int, categories_prompt: str
    ) -> dict:
        """Classify one caller-provided sentence with a sentence-only CoT prompt."""
        prompt_path = self.prompts_dir / "classify_sentence_cot.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read().strip()
        prompt = prompt_template.format(
            categories_prompt=categories_prompt,
            sentence=sentence,
        )
        response = self.generate_response(
            prompt,
            max_new_tokens=16,
            temperature=self.default_temperature,
        )
        try:
            code = self._category_code_from_plain_response(response, sentence_id)
            normalized = self._normalize_sentence_code(code, sentence_id)
            return {
                "has_hate_speech": normalized["has_hate_speech"],
                "sentences": [normalized],
                "raw": response,
            }
        except (TypeError, ValueError, KeyError) as exc:
            return {"sentences": [], "raw": response, "parse_error": str(exc)}

    def classify_sentences_cot(
        self, sentences: list[str], categories_prompt: str
    ) -> dict:
        """Classify fixed sentences one model call at a time, preserving the runner output shape."""
        normalized = []
        raw_outputs = []
        sentence_errors = []
        for sentence_id, sentence in enumerate(sentences, start=1):
            result = self.classify_sentence_cot(
                sentence, sentence_id, categories_prompt
            )
            raw_line = f"{sentence_id}: {result.get('raw', '')}"
            if result.get("parse_error"):
                raw_line += f" [parse_error: {result['parse_error']}]"
            raw_outputs.append(raw_line)
            if result.get("parse_error"):
                sentence_errors.append({
                    "id": sentence_id,
                    "parse_error": result["parse_error"],
                })
                continue
            normalized.extend(result.get("sentences", []))

        return {
            "has_hate_speech": any(item["has_hate_speech"] for item in normalized),
            "sentences": normalized,
            "raw": "\n".join(raw_outputs),
            "sentence_parse_errors": sentence_errors,
        }
