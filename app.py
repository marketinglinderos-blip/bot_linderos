import os
import json
import re
from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import PlainTextResponse, JSONResponse
import requests

# ===================== ENV =====================
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "changeme-verify")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Modelo local GGUF por defecto
GGUF_PATH = os.getenv("GGUF_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
CTX_WINDOW = int(os.getenv("CTX_WINDOW", "4096"))
TOP_K = int(os.getenv("TOP_K", "24"))
TOP_CTX = int(os.getenv("TOP_CTX", "6"))

# Alternar a Together.ai sin tocar el código
# Alternar a Together.ai sin tocar el código
USE_TOGETHER = os.getenv("USE_TOGETHER", "true").lower() == "true"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "a2b3a10dd0c63010b73d07d5a10be3ac8c994434a34ed95cefe5096ffa87cfdd")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

# (Opcional pero recomendable)
if USE_TOGETHER and not TOGETHER_API_KEY:
    print("[WARN] USE_TOGETHER=true pero TOGETHER_API_KEY está vacío. Las llamadas a Together van a fallar.")

# ===================== LLM =====================
LLM = None  # type: ignore

def load_llm():
    """Inicializa el LLM local (o marca Together como proveedor externo)."""
    global LLM
    if LLM is not None:
        return
    if USE_TOGETHER:
        LLM = "together-proxy"
        return
    
# ===================== MANEJO DE SESIONES ========================

import time
from typing import Dict, Any, Optional

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.pending_questions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hora
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Obtiene o crea una sesión"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "turn_count": 0,
                "context": {}
            }
        else:
            self.sessions[session_id]["last_activity"] = time.time()
        return self.sessions[session_id]
    
    def set_pending_question(self, session_id: str, question: str, original_intent: str = None):
        """Establece una pregunta pendiente para la sesión"""
        self.pending_questions[session_id] = {
            "question": question,
            "timestamp": time.time(),
            "original_intent": original_intent
        }
        # Actualizar última actividad
        self.get_session(session_id)
    
    def get_pending_question(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene la pregunta pendiente si existe y no ha expirado"""
        if session_id not in self.pending_questions:
            print(f"🔍 No pending question found for {session_id}")
            return None
        
        pending_q = self.pending_questions[session_id]
        current_time = time.time()
        time_diff = current_time - pending_q["timestamp"]
        
        if time_diff > self.session_timeout:
            print(f"🔍 Pending question expired for {session_id} ({time_diff:.0f}s)")
            del self.pending_questions[session_id]
            return None
        
        print(f"🔍 Found pending question for {session_id}: {pending_q['question']}")
        return pending_q
    
    def clear_pending_question(self, session_id: str):
        """Limpia la pregunta pendiente"""
        if session_id in self.pending_questions:
            del self.pending_questions[session_id]
    
    def has_pending_question(self, session_id: str) -> bool:
        """Verifica si hay pregunta pendiente válida"""
        return self.get_pending_question(session_id) is not None
    
    def cleanup_expired_sessions(self):
        """Limpia sesiones expiradas"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session["last_activity"] > self.session_timeout
        ]
        for sid in expired_sessions:
            del self.sessions[sid]
            if sid in self.pending_questions:
                del self.pending_questions[sid]

# Reemplazar la variable global
SESSION_MANAGER = SessionManager()

# ===================== EMBEDDINGS + FAISS (RAG) =====================
# ===================== EMBEDDINGS (ligero vía API) =====================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large-instruct"  # Modelo más confiable y rápido
)

EMB_DIM = None  # la definimos después del primer llamado

import time
import httpx
from typing import List

def embed(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """Embeddings con backoff exponencial y manejo robusto de errores"""
    global EMB_DIM

    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY está vacío")

    url = "https://api.together.xyz/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "User-Agent": "LinderosBot/1.0"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Backoff exponencial: 2, 4, 8, 16, 32 segundos
            wait_time = 2 ** attempt
            if attempt > 0:
                print(f"[emb] Reintento {attempt}/{max_retries} en {wait_time}s...")
                time.sleep(wait_time)

            with httpx.Client(timeout=60.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                
                data = r.json().get("data", [])
                if not data:
                    raise RuntimeError("API no devolvió datos")

                vectors = [item["embedding"] for item in data]

                if EMB_DIM is None:
                    EMB_DIM = len(vectors[0])
                    print(f"[emb] Dimensión: {EMB_DIM}")

                return vectors

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in [503, 429, 500]:  # Servicio no disponible, rate limit, error interno
                print(f"[emb] Error HTTP {e.response.status_code}, reintentando...")
                continue
            else:
                print(f"[emb] Error HTTP permanente {e.response.status_code}")
                break
                
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exception = e
            print(f"[emb] Error de conexión, reintentando...")
            continue
            
        except Exception as e:
            last_exception = e
            print(f"[emb] Error inesperado: {e}")
            break

    # Si llegamos aquí, todos los reintentos fallaron
    error_msg = f"Falló después de {max_retries} intentos: {last_exception}"
    print(f"[emb] {error_msg}")
    raise RuntimeError(error_msg)

import faiss
import numpy as np
from pathlib import Path
from pypdf import PdfReader

INDEX = None
VEC_IDS: List[str] = []
CHUNK_MAP: Dict[str, Dict[str, Any]] = {}

# ====== RERANKER (cross-encoder) ======
RERANKER = None

def load_reranker():
    global RERANKER
    if RERANKER is not None:
        return
    if not USE_RERANKER:
        return
    try:
        from sentence_transformers import CrossEncoder
        RERANKER = CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        print("[reranker] no se pudo cargar:", repr(e))

# ===================== APP =====================
app = FastAPI(title="Linderos Ucú Bot • Llama (local/Together) + RAG")

BASE_FACTS = {
    "price_list": 200000,
    "down_pct": 0.20,
    "max_months": 36,
    "lot_size_m2": 500,
    "delivery_months": 36,
    "location": "Ucú, Yucatán",
}

SESSIONS: Dict[str, Dict[str, Any]] = {}

# ===================== HELPERS =====================

def get_index() -> faiss.IndexFlatIP:
    global INDEX, EMB_DIM
    if INDEX is None:
        if EMB_DIM is None:
            # Forzamos un embedding mínimo para conocer la dimensión
            _ = embed(["hola"])
        INDEX = faiss.IndexFlatIP(EMB_DIM)
    return INDEX

def pesos(n: float) -> str:
    return f"${n:,.0f} MXN"

def calc_plan() -> Dict[str, Any]:
    p = BASE_FACTS
    down = round(p["price_list"] * p["down_pct"])
    monthly = round((p["price_list"] - down) / p["max_months"])
    return {"price": p["price_list"], "down": down, "months": p["max_months"], "monthly": monthly}

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join((text or "").split())
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i+size]
        if chunk:
            chunks.append(chunk)
        i += max(size - overlap, 1)
    return chunks

def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def reset_index():
    global INDEX, VEC_IDS, CHUNK_MAP
    INDEX = None
    VEC_IDS = []
    CHUNK_MAP = {}

import regex as re

QNA_RE = re.compile(
    r"(?:^|\n)Q:\s*(?P<q>.*?)\nA:\s*(?P<a>.*?)(?=\nQ:|\Z)",
    re.DOTALL | re.IGNORECASE
)

# ===================== SISTEMA DE CONVERSACIÓN MEJORADO =====================
import httpx, time, random, difflib

# Patrones de intención mejorados
INTENT_PATTERNS = {
    "build": r"\b(construir|construcci[oó]n|casa|negocio|edificar|obra|construy[eo]|habitaci[oó]n)\b",
    "price": r"\b(precio|cu[aá]nto|cost[oa]|mensualidad|enganche|pago|plan|dinero|caro|costoso)\b",
    "visit": r"\b(visita|agendar|ir a ver|recorrer|tour|conocer|presencial|cita|reuni[oó]n)\b",
    "location": r"\b(d[oó]nde|ubicaci[oó]n|mapa|maps|direcci[oó]n|distancia|lugar)\b",
    "stock": r"\b(disponible|disponibilidad|lote[s]?|quedan|medidas|500\s*m|m2|tama[oñ]o)\b",
    "investment": r"\b(invertir|inversi[oó]n|plusval[ií]a|renta|rentabilidad|ganancia)\b",
    "objection": r"\b(caro|costoso|carísimo|muy caro|elevado|alto precio)\b",
}

def detect_intent(user_text: str) -> str:
    """Detecta la intención principal del usuario"""
    t = user_text.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, t):
            return intent
    return "fallback"

def handle_price_objection(user_text: str, facts: List[str]) -> str:
    """Maneja objeciones de precio con comparativas y planes"""
    plan = calc_plan()
    
    response = "Entiendo tu consideración sobre el precio. "
    response += f"El terreno de 500m² en Ucú tiene un valor de {pesos(plan['price'])}. "
    response += f"Con un enganche de {pesos(plan['down'])}, quedan {pesos(plan['monthly'])} mensuales por {plan['months']} meses. "
    response += "Comparado con otras zonas en crecimiento, Ucú ofrece una excelente relación costo-beneficio. "
    
    return response

def handle_visit_scheduling() -> str:
    """Ofrece opciones de visita"""
    return "¿Qué día te viene mejor para visitar los terrenos?"

def compose_business_messages(user_text: str, facts: List[str], context_chunks: List[str], intent: str, include_followup: bool = False) -> List[Dict[str, str]]:
    """Sistema de mensajes actualizado para permitir preguntas de seguimiento"""
    
    system_prompt = """Eres un asistente informativo especializado en terrenos en Ucú, Yucatán.

REGLAS ESTRICTAS:
1. DATOS CIERTOS: Usa SOLO información del RAG. Si no hay dato, dilo claramente
2. FORMATO: Mínimo 4 líneas, Máximo 6 líneas, tono cálido y profesional, con emojis
3. PRECIOS: Habla en pesos MXN, NO en porcentajes
4. OBJECIÓN CARO: Ofrece comparativa breve + plan a plazos si aplica
5. UBICACIONES: SOLO Ucú - NO menciones otros lugares
6. EVITA PREGUNTAS: NO hagas preguntas al usuario en tu respuesta

OBJETIVO: Proporcionar información clara y útil, e invitar a profundizar"""

    # Crear contexto estructurado
    context_block = "INFORMACIÓN AUTORIZADA (usa solo esto):\n"
    if facts:
        context_block += "• " + "\n• ".join(facts)
    else:
        context_block += "• No hay información específica sobre este tema"
    
    if context_chunks:
        context_block += "\n\nCONTEXTO ADICIONAL:\n" + "\n".join([f"- {chunk[:150]}..." for chunk in context_chunks[:2]])
    
    full_system = system_prompt + "\n\n" + context_block
    
    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_text}
    ]

def improved_llm_generate(user_text: str, facts: List[str], context_chunks: List[str], intent: str = None, include_followup: bool = False) -> str:
    """Generación con soporte para preguntas de seguimiento"""
    load_llm()
    
    # Manejar casos específicos primero
    if intent == "objection" and any(word in user_text.lower() for word in ["caro", "costoso", "carísimo"]):
        response = handle_price_objection(user_text, facts)
        return response
    
    if intent == "visit":
        visit_response = "¡Excelente! Me encanta que quieras conocer los terrenos. " 
        if facts:
            visit_response += " ".join(facts[:2]) + " "
        visit_response += handle_visit_scheduling()
        return visit_response
    
    # Pasar include_followup al composer
    messages = compose_business_messages(user_text, facts, context_chunks, intent, include_followup)
    
    if USE_TOGETHER:
        url = "https://api.together.xyz/v1/chat/completions"
        payload = {
            "model": TOGETHER_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 180,  # Aumentar tokens para incluir pregunta
            "stop": ["</s>", "Q:", "A:", "Fuente:", "Referencia:"],
        }
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return apply_business_cleanup(txt, user_text, facts, intent)
        except Exception as e:
            print(f"[Together.ai error] {e}")
            return business_fallback(user_text, facts, intent)
    
    # Fallback local
    try:
        out = LLM.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.8,
            max_tokens=180,  # Aumentar tokens
            stop=["</s>", "Q:", "A:", "Fuente:", "Referencia:"]
        )
        txt = (out["choices"][0]["message"]["content"] or "").strip()
        return apply_business_cleanup(txt, user_text, facts, intent)
    except Exception as e:
        print(f"[Local LLM error] {e}")
        return business_fallback(user_text, facts, intent)

def apply_business_cleanup(text: str, user_text: str, facts: List[str], intent: str = None) -> str:
    """Limpieza con reglas de negocio"""
    if not text:
        return business_fallback(user_text, facts, intent)
    
    # Limpiar etiquetas Q/A
    text = re.sub(r'\b[QA]\s*:\s*', '', text)
    
    # Verificar longitud
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 6:
        text = '\n'.join(lines[:6])
    
    return text

def business_fallback(user_text: str, facts: List[str], intent: str = None) -> str:
    """Fallback con reglas de negocio"""
    if not facts:
        return "No tengo información específica sobre eso. ¿En qué más puedo ayudarte?"
    
    # Usar los hechos disponibles
    response = "Según la información disponible: " + ". ".join(facts[:2])
    
    return response

PENDING_QUESTIONS: Dict[str, Dict[str, Any]] = {}

def generate_followup_question(user_text: str, context: str) -> str:
    """Genera pregunta que invita al usuario a pedir más información específica"""
    print(f"🔍 Generating followup for: '{user_text}' with context: '{context}'")
    
    prompt = f"""
Basado en esta conversación:
Usuario: "{user_text}"
Contexto: {context}

Genera UNA sola pregunta que invite al usuario a SOLICITAR MÁS INFORMACIÓN ESPECÍFICA sobre los terrenos.

La pregunta debe:
- Ser directa y concreta
- Invitar a pedir información, no opiniones
- Referirse a temas específicos de terrenos
- Ser fácil de responder pidiendo datos
- EVITAR preguntar sobre "tamaño y precios" (ya se mencionó)

EJEMPLOS VARIADOS:
- "¿Te gustaría que te comparta la ubicación exacta en Google Maps?"
- "¿Quieres que te explique más sobre nuestros planes de financiamiento a 36 meses?"
- "¿Necesitas conocer los detalles sobre el proceso de escrituración?"
- "¿Te interesa saber más sobre la plusvalía del 35% anual en Ucú?"
- "¿Quieres que te comparta los testimonios de clientes que ya construyeron?"
- "¿Te gustaría agendar una visita para conocer los terrenos en persona?"
- "¿Necesitas más información sobre los permisos de construcción?"
- "¿Quieres conocer los detalles del kit legal y registros ante INSEJUPY?"
- "¿Te interesa saber sobre las opciones de pago desde $5,000 MXN de enganche?"
- "¿Quieres conocer los planes de desarrollo urbano alrededor de los terrenos?"

IMPORTANTE: 
- Devuelve SOLO la pregunta, sin explicaciones adicionales.
- NO repitas "tamaño y precios"
- Basado en el contexto, elige un tema diferente

PREGUNTA QUE INVITA A PEDIR INFORMACIÓN:
"""
    
    if USE_TOGETHER:
        messages = [{"role": "user", "content": prompt}]
        url = "https://api.together.xyz/v1/chat/completions"
        payload = {
            "model": TOGETHER_MODEL,
            "messages": messages,
            "temperature": 0.5,  # Más variación para preguntas diferentes
            "max_tokens": 45,
            "stop": ["\n\n", "Usuario:", "Contexto:"]
        }
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            result = r.json()
            question = (result["choices"][0]["message"]["content"] or "").strip()
            
            # Limpiar y formatear la pregunta
            question = re.sub(r'^["\']|["\']$', '', question)  # Remover comillas
            question = question.rstrip('.').strip()
            
            # Asegurar que termina con ?
            if not question.endswith('?'):
                question += '?'
                
            print(f"🔍 Generated question: '{question}'")
            
            # Validación mínima
            if len(question) < 10 or question == '?':
                return get_random_fallback_question()
                
            return question
            
        except Exception as e:
            print(f"🔍❌ Error generating followup: {e}")
            return get_random_fallback_question()
    else:
        return get_random_fallback_question()

def get_random_fallback_question() -> str:
    """Preguntas de respaldo variadas"""
    fallback_questions = [
        "¿Te gustaría que te comparta la ubicación exacta en Google Maps?",
        "¿Quieres que te explique más sobre nuestros planes de financiamiento a 36 meses?",
        "¿Necesitas conocer los detalles sobre el proceso de escrituración?",
        "¿Te interesa saber más sobre la plusvalía en Ucú?",
        "¿Quieres agendar una visita para conocer los terrenos en persona?",
        "¿Necesitas más información sobre los permisos de construcción?",
        "¿Te gustaría conocer los testimonios de clientes que ya construyeron?",
        "¿Quieres saber sobre las opciones de pago desde $5,000 MXN de enganche?"
    ]
    return random.choice(fallback_questions)

def rewrite_user_response(user_text: str, original_question: str) -> str:
    """Reescribe la respuesta del usuario como prompt para continuar"""
    prompt = f"""
Pregunta original: "{original_question}"
Respuesta del usuario: "{user_text}"

Reescribe esto como una consulta clara para buscar información sobre terrenos. 
Incluye el contexto completo de la conversación.

Ejemplo:
- Si pregunta fue "¿Para construcción o inversión?" y usuario dice "construir mi casa"
- Output: "Información sobre terrenos para construcción de casa habitación"

CONSULTA REWRITE:"""
    
    if USE_TOGETHER:
        messages = [{"role": "user", "content": prompt}]
        url = "https://api.together.xyz/v1/chat/completions"
        payload = {
            "model": TOGETHER_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 40
        }
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            rewritten = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return rewritten
        except Exception:
            return user_text

def is_positive_response(text: str) -> bool:
    """Detecta si la respuesta es positiva"""
    positive_patterns = [
        r'\b(s[ií]|claro|por supuesto|ok|vale|perfecto|adelante|s[ií] por favor|por favor|obvio|desde luego|por qué no|seguro|bueno|está bien|de acuerdo|acepto|con gusto)\b',
        r'👍|✅|😊|🙂|🥰',
    ]
    
    text_lower = text.lower().strip()
    
    # Respuestas muy cortas positivas
    if text_lower in ['sí', 'si', 's', 'y', 'yes', 'ok', 'vale']:
        return True
        
    # Patrones regex
    for pattern in positive_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
            
    return False

def is_negative_response(text: str) -> bool:
    """Detecta si la respuesta es negativa"""
    negative_patterns = [
        r'\b(no|nah|nop|nada|para nada|negativo|ni hablar|olv[ií]dalo|cancelar|detente|basta|ya no|no quiero)\b',
        r'👎|❌|😞|🙁',
    ]
    
    text_lower = text.lower().strip()
    
    if text_lower in ['no', 'n', 'nop', 'nah']:
        return True
        
    for pattern in negative_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
            
    return False

# ===================== BÚSQUEDA Y RAG =====================
def search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    print(f"[search] Buscando: '{query}'")
    print(f"[search] VEC_IDS disponibles: {len(VEC_IDS)}")
    
    # Si no hay vectores, no hay nada que buscar
    if len(VEC_IDS) == 0:
        print("[search] ❌ No hay vectores indexados")
        return []

    # Aseguramos tener índice
    idx = get_index()
    if idx.ntotal == 0:
        print("[search] ❌ Índice FAISS vacío")
        return []

    print(f"[search] Índice FAISS tiene {idx.ntotal} vectores")
    
    try:
        Q = np.array(embed([query]), dtype=np.float32)
        faiss.normalize_L2(Q)
        D, I = idx.search(Q, top_k)
        print(f"[search] Búsqueda FAISS completada, resultados: {len(I[0])}")
        
        hits = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            vec_id = VEC_IDS[i]
            meta = CHUNK_MAP.get(vec_id, {})
            meta = {**meta, "score": float(score), "id": vec_id}
            hits.append(meta)

        # Filtrar y ordenar
        seen = set()
        filtered = []
        for h in hits:
            key = h.get("text", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            if h.get("kind") == "qna":
                h["score"] += 0.2
            filtered.append(h)

        result = sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_k]
        print(f"[search] Resultados filtrados: {len(result)}")
        
        # Mostrar los top 3 resultados
        for i, hit in enumerate(result[:3]):
            print(f"[search] Top {i+1}: score={hit['score']:.3f}, text={hit['text'][:100]}...")
            
        return result
        
    except Exception as e:
        print(f"[search] ERROR en búsqueda: {e}")
        return []

def rerank(query: str, hits: List[Dict[str, Any]], top_n: int = 6) -> List[Dict[str, Any]]:
    if not hits or RERANKER is None:
        return hits[:top_n]
    pairs = [(query, h["text"]) for h in hits]
    try:
        scores = RERANKER.predict(pairs)
    except Exception as e:
        print("[rerank] error:", repr(e))
        return hits[:top_n]
    for h, s in zip(hits, scores):
        h["_rerank_score"] = float(s)
    hits_sorted = sorted(hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return hits_sorted[:top_n]

def extract_facts_from_matches(matches: List[Dict[str, Any]]) -> List[str]:
    """Extrae hechos clave"""
    facts = []
    for match in matches:
        if "a" in match:
            answer = match["a"]
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            for sent in sentences:
                sent = re.sub(r'\s+', ' ', sent).strip()
                if len(sent) > 10 and not sent.lower().startswith(('q:', 'a:')):
                    facts.append(sent)
    return facts[:8]

def find_best_qna_matches(user_text: str, hits: List[Dict[str, Any]], max_matches: int = 3) -> List[Dict[str, Any]]:
    """Encuentra mejores matches Q/A"""
    matches = []
    ut = re.sub(r'\s+', ' ', user_text.lower()).strip()
    
    for h in hits:
        txt = h.get("text", "")
        m = re.search(r"Q:\s*(.+?)\nA:\s*(.+)$", txt, flags=re.I | re.S)
        if not m:
            continue
            
        q = re.sub(r'\s+', ' ', m.group(1).lower()).strip()
        a = m.group(2).strip()
        
        semantic_score = difflib.SequenceMatcher(None, ut, q).ratio()
        user_words = set(ut.split())
        q_words = set(q.split())
        common_words = user_words.intersection(q_words)
        keyword_bonus = len(common_words) * 0.05 if common_words else 0
        
        total_score = semantic_score + keyword_bonus
        
        if total_score >= 0.4:
            matches.append({
                "q": m.group(1).strip(),
                "a": a,
                "h": h,
                "score": total_score,
            })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:max_matches]

# ===================== INDEXACIÓN =====================
def add_to_index(doc_id: str, title: str, text: str):
    """Añade documento al índice usando embeddings vía API y un índice FAISS perezoso."""
    # Aseguramos que el índice exista
    idx = get_index()

    qna_chunks = parse_qna_chunks(text)
    if qna_chunks:
        vecs = embed(qna_chunks)
        X = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(X)
        idx.add(X)
        for j, c in enumerate(qna_chunks):
            vec_id = f"{doc_id}::qna::{j}"
            VEC_IDS.append(vec_id)
            CHUNK_MAP[vec_id] = {
                "doc_id": doc_id,
                "title": title,
                "text": c,
                "kind": "qna",
            }
        return

    chunks = chunk_text(text)
    if not chunks:
        return

    vecs = embed(chunks)
    X = np.array(vecs, dtype=np.float32)
    faiss.normalize_L2(X)
    idx.add(X)
    for j, c in enumerate(chunks):
        vec_id = f"{doc_id}::{j}"
        VEC_IDS.append(vec_id)
        CHUNK_MAP[vec_id] = {
            "doc_id": doc_id,
            "title": title,
            "text": c,
            "kind": "raw",
        }

def parse_qna_chunks(raw: str) -> List[str]:
    """Extrae Q/A del texto"""
    text = raw.replace("\r", "").strip()
    chunks = []
    
    # Patrón mejorado para Q/A
    qna_pattern = re.compile(
        r'Q:\s*(.*?)\s*A:\s*(.*?)(?=(?:\s*Q:\s*|\s*$))',
        re.DOTALL | re.IGNORECASE
    )
    
    matches = qna_pattern.findall(text)
    
    for q, a in matches:
        q_clean = re.sub(r'\s+', ' ', q).strip()
        a_clean = re.sub(r'\s+', ' ', a).strip()
        
        if (q_clean and a_clean and 
            len(q_clean) > 5 and len(a_clean) > 5):
            
            chunks.append(f"Q: {q_clean}\nA: {a_clean}")
    
    return chunks

# ===================== SALUDOS Y MENSAJES INICIALES =====================

def first_turn_message() -> str:
    plan = calc_plan()
    return (
        "¡Hola! 👋 Soy tu asistente de Terrenos Mérida MX. "
        f"Tenemos terrenos en Ucú desde {pesos(plan['price'])} por lote de 500 m². "
        f"Enganche desde {pesos(plan['down'])}.\n"
        "¿En qué puedo ayudarte hoy?"
    )

# ===================== PROCESADOR PRINCIPAL =====================
def process_user_message(user_text: str, session_id: str = "default") -> str:
    # Limpiar sesiones expiradas periódicamente
    if random.random() < 0.1:
        SESSION_MANAGER.cleanup_expired_sessions()
    
    session = SESSION_MANAGER.get_session(session_id)
    session["turn_count"] += 1
    
    print(f"=== DEBUG SESSION {session_id} ===")
    print(f"User: '{user_text}'")
    print(f"Pending Q: {SESSION_MANAGER.get_pending_question(session_id)}")

    # Verificar si hay pregunta pendiente
    pending_q = SESSION_MANAGER.get_pending_question(session_id)
    
    if pending_q:
        print(f"🔍 Hay pregunta pendiente: {pending_q['question']}")
        
        # ANALIZAR SI EL USUARIO QUIERE RESPONDER LA PREGUNTA O HABLAR DE OTRA COSA
        user_intent = analyze_intent_with_llm(user_text, pending_q['question'])
        print(f"🔍 Intención detectada: {user_intent}")
        
        if user_intent == "responder_pendiente":
            # El usuario quiere responder a la pregunta pendiente
            if is_negative_response(user_text):
                SESSION_MANAGER.clear_pending_question(session_id)
                return "¡Entendido! 😊 ¿En qué más puedo ayudarte?"
            else:
                new_query = rewrite_user_response(user_text, pending_q['question'])
                SESSION_MANAGER.clear_pending_question(session_id)
                return process_normal_flow(new_query, session_id, ask_followup=True)
        
        else:
            # El usuario quiere hablar de otra cosa, ignorar pregunta pendiente
            SESSION_MANAGER.clear_pending_question(session_id)
            return process_normal_flow(user_text, session_id, ask_followup=True)
    
    # Flujo normal (sin pregunta pendiente)
    return process_normal_flow(user_text, session_id, ask_followup=True)

def analyze_intent_with_llm(user_text: str, pending_question: str) -> str:
    """
    Usa el LLM para determinar si el usuario quiere:
    - responder_pendiente: Responder a la pregunta pendiente
    - nuevo_tema: Hablar de algo completamente diferente
    """
    
    prompt = f"""
Analiza si el usuario quiere RESPONDER a la pregunta pendiente o hablar de un NUEVO TEMA.

PREGUNTA PENDIENTE: "{pending_question}"
MENSAJE DEL USUARIO: "{user_text}"

OPCIONES:
- "responder_pendiente": Si el usuario está respondiendo directamente a la pregunta pendiente
- "nuevo_tema": Si el usuario está preguntando sobre algo completamente diferente

Ejemplos:
- Usuario: "sí" → responder_pendiente
- Usuario: "no" → responder_pendiente  
- Usuario: "claro que sí" → responder_pendiente
- Usuario: "qué precios tienen?" → nuevo_tema
- Usuario: "hola" → nuevo_tema
- Usuario: "información sobre terrenos" → nuevo_tema
- Usuario: "sí, quiero info" → responder_pendiente

RESPUESTA (solo una palabra: responder_pendiente o nuevo_tema):
"""
    
    if USE_TOGETHER:
        messages = [{"role": "user", "content": prompt}]
        url = "https://api.together.xyz/v1/chat/completions"
        payload = {
            "model": TOGETHER_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 20,
            "stop": ["\n"]
        }
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()
            intent = (result["choices"][0]["message"]["content"] or "").strip().lower()
            
            # Validar respuesta
            if "responder" in intent:
                return "responder_pendiente"
            elif "nuevo" in intent:
                return "nuevo_tema"
            else:
                # Por defecto, asumir que es nuevo tema si no está claro
                return "nuevo_tema"
                
        except Exception as e:
            print(f"🔍❌ Error analizando intención: {e}")
            # Por defecto, asumir nuevo tema si hay error
            return "nuevo_tema"
    
    # Fallback simple si no hay LLM
    return simple_intent_analysis(user_text)

def simple_intent_analysis(user_text: str) -> str:
    """Análisis simple de intención como fallback"""
    user_lower = user_text.lower().strip()
    
    # Respuestas muy cortas o afirmativas/negativas
    short_responses = ['sí', 'si', 's', 'no', 'n', 'claro', 'por supuesto', 'ok', 'vale']
    if user_lower in short_responses or len(user_text.split()) <= 2:
        return "responder_pendiente"
    
    # Si contiene palabras clave de terrenos, probablemente es nuevo tema
    terreno_keywords = ['terreno', 'lote', 'precio', 'costo', 'cuánto', 'ubicación', 'mapa', 'medida']
    if any(keyword in user_lower for keyword in terreno_keywords):
        return "nuevo_tema"
    
    # Por defecto, asumir nuevo tema
    return "nuevo_tema"

def process_normal_flow(user_text: str, session_id: str, ask_followup: bool = True) -> str:
    """Tu función original + siempre pregunta seguimiento"""
    
    # VERIFICACIÓN MÁS ROBUSTA DE PREGUNTAS PENDIENTES
    pending_q = SESSION_MANAGER.get_pending_question(session_id)
    has_pending = pending_q is not None
    
    print(f"🔍 Pending question detected: {has_pending}")
    if has_pending:
        print(f"🔍 Pending Q: {pending_q['question']}")
    
    # Detectar saludos
    def is_greeting(text: str) -> bool:
        """Versión super simple para testing"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        # Saludos básicos que deberían funcionar
        simple_greetings = [
            'hola', 'holaa', 'holaaa', 'holi',
            'buenas', 'buenos días', 'buen día', 
            'buenas tardes', 'buenas noches',
            'hello', 'hey', 'hi'
        ]
        
        for greeting in simple_greetings:
            if text_lower.startswith(greeting):
                print(f"🔍✅ SALUDO DETECTADO: '{text}' comienza con '{greeting}'")
                return True
        
        print(f"🔍❌ NO ES SALUDO: '{text}'")
        return False
    
    # Detectar intención
    intent = detect_intent(user_text)
    print(f"🔍 Detected intent: {intent}")
    
    # Buscar información relevante
    raw_hits = search(user_text, top_k=TOP_K)
    reranked_hits = rerank(user_text, raw_hits, top_n=TOP_CTX)

    # PARCHE TEMPORAL - Buscar match por keywords específicos
    user_lower = user_text.lower()
    if any(kw in user_lower for kw in ["oficina", "horario", "sucursal", "física"]):
        for hit in reranked_hits:
            if "oficina" in hit.get("text", "").lower():
                # Forzar este hit al top
                reranked_hits.remove(hit)
                reranked_hits.insert(0, hit)
                print(f"🔍✅ OFICINA DETECTADA - Hit movido al top")
                break
    
    # Encontrar mejores matches Q/A
    qna_matches = find_best_qna_matches(user_text, reranked_hits, max_matches=3)
    
    # Extraer contexto
    facts = extract_facts_from_matches(qna_matches)
    context_chunks = [h["text"] for h in reranked_hits[:3]]
    
    # Generar respuesta principal
    main_response = improved_llm_generate(user_text, facts, context_chunks, intent, include_followup=ask_followup)

    print(f"🔍 Main response has followup: {has_followup_question(main_response)}")
    print(f"🔍 Session has pending question: {has_pending}")
    print(f"🔍 Ask followup: {ask_followup}")
    
    # ✅ MOVER ESTO DENTRO DEL IF CORRESPONDIENTE
    # Solo hacer pregunta de seguimiento si no hay una pendiente Y ask_followup=True
    if ask_followup and not has_pending:
        context_for_question = " | ".join(facts[:2]) if facts else user_text
        followup_question = generate_followup_question(user_text, context_for_question)
        print(f"🔍 Generated followup question: {followup_question}")
        
        if followup_question and len(followup_question) > 15:
            SESSION_MANAGER.set_pending_question(session_id, followup_question, intent)
            print(f"🔍✅ SET PENDING QUESTION: {followup_question}")
            # ✅ SIEMPRE agregar la pregunta
            return main_response + f"\n\n{followup_question}"
        else:
            print(f"🔍❌ Followup question invalid: {followup_question}")
    
    return main_response
    

def process_direct_query(query: str, session_id: str) -> str:
    """Procesa una consulta directa sin hacer preguntas de seguimiento"""
    intent = detect_intent(query)
    raw_hits = search(query, top_k=TOP_K)
    reranked_hits = rerank(query, raw_hits, top_n=TOP_CTX)
    qna_matches = find_best_qna_matches(query, reranked_hits, max_matches=3)
    facts = extract_facts_from_matches(qna_matches)
    context_chunks = [h["text"] for h in reranked_hits[:3]]
    
    # Generar respuesta SIN pregunta de seguimiento
    return improved_llm_generate(query, facts, context_chunks, intent, include_followup=False)

def has_followup_question(text: str) -> bool:
    """Detecta si la respuesta ya incluye una pregunta de seguimiento REAL"""
    # Buscar patrones específicos de preguntas de seguimiento
    followup_patterns = [
        r'\b(te gustaría|quieres|necesitas|te interesa|deseas)\b.*\?',
        r'\?.*\b(te gustaría|quieres|necesitas|te interesa|deseas)\b',
        r'¿.*\?',  # Si ya tiene signos de pregunta en español
    ]
    
    text_lower = text.lower()
    
    for pattern in followup_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Contar signos de pregunta
    question_marks = text.count('?') + text.count('¿')
    return question_marks > 1  # Si tiene más de un signo de pregunta, probablemente ya tiene preguntas

# ===================== ENDPOINTS =====================
@app.get("/webhook", response_class=PlainTextResponse)
def verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse("forbidden", status_code=403)

@app.post("/webhook")
async def receive(request: Request):
    body = await request.json()
    try:
        for entry in body.get("entry", []):
            for evt in entry.get("messaging", []):
                psid = evt.get("sender", {}).get("id")
                if not psid:
                    continue
                text = None
                if "message" in evt and "text" in evt["message"]:
                    text = evt["message"]["text"].strip()
                elif "postback" in evt and evt["postback"].get("title"):
                    text = evt["postback"].get("title").strip()
                if not text:
                    continue

                sess = SESSIONS.setdefault(psid, {"turns": 0})
                sess["turns"] += 1

                if sess["turns"] == 1:
                    send_text_psid(psid, first_turn_message())
                    continue

                reply = process_user_message(text, session_id=psid)
                send_text_psid(psid, reply)

    except Exception as e:
        print("[webhook] error:", repr(e))
    return JSONResponse({"status": "ok"})

@app.post("/dev/chat")
async def dev_chat(payload: dict, request: Request):
    text = (payload.get("message") or "").strip()
    session_id = request.headers.get("X-Session-ID", "default")  # ← Del header
    reply = process_user_message(text, session_id)
    return {"reply": reply}

# ===================== ADMIN ENDPOINTS =====================
@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...)):
    try:
        pdf_dir = Path("./pdfs")
        pdf_dir.mkdir(exist_ok=True)

        path = pdf_dir / file.filename
        print(f"[admin_upload] Recibido archivo: {file.filename}")
        with open(path, "wb") as f:
            content = await file.read()
            print(f"[admin_upload] Bytes recibidos: {len(content)}")
            f.write(content)

        print(f"[admin_upload] Leyendo PDF desde: {path}")
        text = load_pdf_text(path)
        print(f"[admin_upload] Texto extraído, longitud: {len(text)}")

        add_to_index(doc_id=path.stem, title=file.filename, text=text)
        print(f"[admin_upload] Index actualizado. Chunks totales: {len(VEC_IDS)}")

        return {
            "status": "ok",
            "docs": len(set(v.split("::")[0] for v in VEC_IDS)),
            "chunks": len(VEC_IDS),
        }

    except Exception as e:
        import traceback
        print("[admin_upload] ERROR:", repr(e))
        traceback.print_exc()
        # Deja que FastAPI responda 500, pero ya tendrás el stacktrace en consola
        raise

@app.post("/admin/reindex")
async def admin_reindex():
    reset_index()
    pdf_dir = Path("./pdfs")
    pdf_dir.mkdir(exist_ok=True)
    count_docs = 0
    for pdf in pdf_dir.glob("**/*.pdf"):
        try:
            text = load_pdf_text(pdf)
            add_to_index(pdf.stem, pdf.name, text)
            count_docs += 1
        except Exception:
            continue
    return {"status": "ok", "docs": count_docs, "chunks": len(VEC_IDS)}

@app.get("/admin/stats")
async def admin_stats():
    # Si nunca se ha usado el índice, INDEX puede ser None
    index_size = 0
    if INDEX is not None:
        index_size = int(INDEX.ntotal)

    return {
        "docs": len(set(v.split("::")[0] for v in VEC_IDS)),
        "chunks": len(VEC_IDS),
        "index_size": index_size,
        "emb_dim": EMB_DIM,
        "ctx_window": CTX_WINDOW,
        "use_together": USE_TOGETHER,
        "model": TOGETHER_MODEL if USE_TOGETHER else GGUF_PATH,
    }

def send_text_psid(psid: str, text: str):
    if not PAGE_ACCESS_TOKEN:
        print(f"→ {psid}: {text}")
        return
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {"recipient": {"id": psid}, "message": {"text": text}}
    try:
        r = requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[send_text_psid] exception:", e)

# ===================== INICIALIZACIÓN =====================
@app.on_event("startup")
async def on_startup():
    load_llm()
#    load_reranker()
    pdf_dir = Path("./pdfs")
    pdf_dir.mkdir(exist_ok=True)
    if any(pdf_dir.glob("**/*.pdf")):
        await admin_reindex()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
