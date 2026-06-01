#!/usr/bin/env python3
"""
INFERA — build_prompt_dataset.py
Construye el corpus de 90 prompts para el benchmark.

DISEÑO:
  Tres casos operacionalizan los niveles de VI4 (carga contextual efectiva):
    Caso A (~256 tokens): chatbot de atención al cliente, contexto corto
    Caso B (~1024 tokens): asistente conversacional con historial
    Caso C (~4096 tokens): análisis de documentos completos

DECISIÓN TÉCNICA CLAVE — truncación token-aware:
  El parche anterior usaba slices de caracteres (profile[:400]) como
  aproximación de tokens. Esto es frágil: los archivos de contexto pueden
  cambiar y los conteos de caracteres NO garantizan conteos de tokens.
  Este script usa tokenizer.encode() para truncar exactamente por tokens
  y tokenizer.apply_chat_template() para contar lo que vLLM realmente ve,
  incluyendo los tokens especiales de plantilla.

VALIDACIÓN:
  Tolerancia VI4: ±15% del objetivo central.
    Case A: 256 tokens → válido en [218, 294]
    Case B: 1024 tokens → válido en [870, 1178]
    Case C: 4096 tokens → válido en [3482, 4710]
  Si algún caso produce < 30 prompts válidos → el script aborta con error.
  No hay fallbacks silenciosos.

REPRODUCIBILIDAD:
  - Las 30 preguntas de usuario por caso están embebidas en este script.
  - El corpus generado es idéntico en cualquier máquina con los mismos
    archivos de contexto y este script.
  - Tokenizador: meta-llama/Meta-Llama-3.1-8B-Instruct (real, no aproximado).

USO:
  # Verificar token counts sin guardar:
  python scripts/build_prompt_dataset.py --verify-only

  # Construir corpus completo:
  python scripts/build_prompt_dataset.py

  # Especificar rutas si difieren de los defaults:
  python scripts/build_prompt_dataset.py \\
      --model-dir /workspace/models/llama3.1-8b-instruct \\
      --context-dir data/context \\
      --conversations data/conversations/conversation_histories.jsonl \\
      --output data/prompts/prompt_corpus.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — targets y tolerancias de VI4
# ─────────────────────────────────────────────────────────────────────────────

VI4_TARGETS = {
    "A": {"center": 256,  "low": 218,  "high": 294},
    "B": {"center": 1024, "low": 870,  "high": 1178},
    "C": {"center": 4096, "low": 3482, "high": 4710},
}

PROMPTS_PER_CASE = 30
TOLERANCE = 0.15  # ±15%

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTAS DE USUARIO
# 30 por caso. Embebidas aquí para garantizar reproducibilidad.
# En español (idioma del caso de uso objetivo).
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS_A = [
    "¿Cuál es el horario de atención al cliente?",
    "¿Cómo puedo reportar un problema técnico urgente?",
    "¿Cuáles son los métodos de pago aceptados para la renovación?",
    "¿Cuánto tiempo tarda en resolverse una incidencia de nivel crítico?",
    "¿Qué incluye el plan de mantenimiento preventivo básico?",
    "¿Puedo solicitar un cambio de plan sin penalización?",
    "¿Qué documentos necesito para abrir un ticket de soporte?",
    "¿Cómo se calcula el tiempo de respuesta garantizado en mi contrato?",
    "¿Existe algún número de emergencia disponible las 24 horas?",
    "¿Qué pasa si el técnico no llega en el tiempo comprometido?",
    "¿Puedo solicitar una copia de mi historial de incidencias?",
    "¿Cómo actualizo los datos de contacto de mi empresa en el sistema?",
    "¿Qué significa el nivel de servicio Silver en la política de atención?",
    "¿Cuántos usuarios pueden estar registrados bajo una sola licencia corporativa?",
    "¿Existe penalización por cancelar el servicio antes de que termine el contrato?",
    "¿Cómo funciona el proceso de escalamiento de un caso no resuelto?",
    "¿Qué cobertura tiene el soporte remoto versus el soporte presencial?",
    "¿Puedo transferir mi contrato de servicio a otra empresa?",
    "¿Cómo solicito capacitación técnica para mi equipo?",
    "¿Qué pasa con mis datos si decido cancelar el servicio?",
    "¿Con qué frecuencia se realizan las actualizaciones del sistema?",
    "¿El soporte cubre problemas con equipos de terceros integrados al sistema?",
    "¿Cómo reporto un incumplimiento del acuerdo de nivel de servicio?",
    "¿Puedo añadir sedes adicionales a mi contrato actual?",
    "¿Cuál es el proceso para solicitar un reembolso por tiempo de inactividad?",
    "¿Qué información necesito tener a mano antes de llamar al soporte?",
    "¿Existe algún portal de autogestión donde pueda ver el estado de mis tickets?",
    "¿Cómo se define una interrupción parcial versus una interrupción total del servicio?",
    "¿El mantenimiento programado se notifica con anticipación?",
    "¿Qué garantías de seguridad de datos ofrece la empresa?",
]

QUESTIONS_B = [
    "¿Cuándo recibiré la confirmación oficial del cierre de mi caso?",
    "¿Existe algún costo adicional asociado al soporte que recibí?",
    "¿Puedo solicitar que el mismo técnico atienda mi próxima incidencia?",
    "¿Cuál es el siguiente paso para formalizar lo que acordamos?",
    "¿Hay alguna documentación que deba firmar para continuar el proceso?",
    "¿En qué plazo se implementará la solución que discutimos?",
    "¿Puedo obtener un resumen escrito de lo conversado hasta ahora?",
    "¿Esto que mencionas implica algún cambio en mi plan de servicio actual?",
    "¿Necesito coordinar algo con mi equipo interno para que esto funcione?",
    "¿Existe algún riesgo de interrupción durante la implementación que comentaste?",
    "¿El cambio que me propones está cubierto por mi contrato vigente?",
    "¿Con quién más debo hablar en tu empresa para avanzar con esto?",
    "¿Puedo recibir una alerta anticipada si el problema podría volver a ocurrir?",
    "¿Qué indicadores debo monitorear para saber que todo está funcionando bien?",
    "¿Hay algo que yo deba evitar hacer para que la solución sea estable?",
    "¿Cuánto tiempo tomará el proceso de validación que mencionaste?",
    "¿Eso que describen aplica también a las sucursales que tengo fuera de Guayaquil?",
    "¿Puedo cancelar el cambio si no estoy satisfecho con los resultados?",
    "¿Hay algún periodo de prueba antes de que los cambios sean definitivos?",
    "¿Puedo revisar el informe técnico antes de que se archive el caso?",
    "¿Eso que mencionan afecta a todos los usuarios o solo al administrador principal?",
    "¿Qué pasa si el problema que describieron vuelve a ocurrir en menos de 30 días?",
    "¿Necesito actualizar alguna configuración de mi parte para que esto funcione?",
    "¿El soporte que recibiré en adelante cambia en algo respecto a lo actual?",
    "¿Pueden enviarme la propuesta técnica por escrito para revisarla con mi gerente?",
    "¿Cuál es el tiempo estimado de inactividad durante el mantenimiento que proponen?",
    "¿Esto tiene algún impacto en mis respaldos automáticos programados?",
    "¿Puedo solicitar que esto sea revisado por un ingeniero sénior antes de aplicarlo?",
    "¿Hay algún formulario que deba completar para autorizar los cambios?",
    "¿Eso que mencionan ya fue probado en entornos similares al mío?",
]

QUESTIONS_C = [
    "¿Cuáles son las principales obligaciones del cliente según este documento?",
    "Resume las cláusulas relacionadas con la terminación anticipada del contrato.",
    "¿Qué garantías específicas ofrece la empresa en este acuerdo?",
    "¿Cuáles son los criterios para determinar incumplimiento por parte del proveedor?",
    "¿Qué mecanismos de resolución de conflictos se establecen en el documento?",
    "¿Cuáles son las limitaciones de responsabilidad del proveedor descritas aquí?",
    "¿Qué información confidencial está protegida bajo este acuerdo y cómo?",
    "Identifica las condiciones bajo las cuales se puede suspender el servicio.",
    "¿Cuáles son los plazos de pago y las penalizaciones por retraso establecidos?",
    "¿Qué modificaciones al servicio requieren autorización formal según este documento?",
    "¿Cuáles son las condiciones de renovación automática del contrato?",
    "¿Qué niveles de servicio (SLA) están comprometidos y cómo se miden?",
    "¿Cuáles son los derechos de propiedad intelectual descritos en este acuerdo?",
    "Identifica todos los plazos críticos mencionados en el documento.",
    "¿Qué exclusiones de servicio están definidas explícitamente?",
    "¿Cuáles son las condiciones para que el cliente pueda reclamar compensaciones?",
    "¿Qué documentación debe entregarse al inicio y al cierre del contrato?",
    "¿Cuáles son las obligaciones de seguridad de la información del proveedor?",
    "¿Bajo qué circunstancias puede una parte ceder sus derechos a un tercero?",
    "¿Cuáles son los procedimientos de escalamiento establecidos en este documento?",
    "¿Qué define el documento como 'fuerza mayor' y cuáles son sus consecuencias?",
    "¿Cuáles son las condiciones bajo las cuales se aplican descuentos o créditos?",
    "Identifica las cláusulas relacionadas con auditorías o revisiones del servicio.",
    "¿Qué métricas se usan para medir el cumplimiento del nivel de servicio?",
    "¿Cuáles son las restricciones de uso del servicio mencionadas en el documento?",
    "¿Qué obligaciones tiene el proveedor en caso de incidente de seguridad?",
    "¿Cuáles son las condiciones para modificar el precio del servicio durante el contrato?",
    "¿Qué garantías de continuidad del negocio ofrece el proveedor en este acuerdo?",
    "Identifica todas las partes mencionadas en el documento y sus roles.",
    "¿Cuáles son los criterios de aceptación del servicio establecidos?",
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE TOKENIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_dir: str):
    """
    Carga el tokenizador desde el directorio local del modelo.
    Falla con mensaje claro si no está disponible.

    Por qué cargamos local y no desde HuggingFace:
      En RunPod sin internet, el modelo ya está en /workspace/models/.
      En reproducibilidad, garantizamos que todos usen exactamente el mismo
      tokenizador que se usó para el experimento.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("FATAL: transformers no instalado. Ejecuta el setup primero.", file=sys.stderr)
        sys.exit(1)

    model_path = Path(model_dir)
    if not model_path.exists():
        # Intentar también desde HuggingFace cache como fallback
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"  Directorio local no encontrado: {model_path}", file=sys.stderr)
        print(f"  Intentando cargar desde HuggingFace: {model_id}", file=sys.stderr)
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            print(f"  Tokenizador cargado desde HuggingFace cache.")
            return tok
        except Exception as e:
            print(f"FATAL: No se pudo cargar el tokenizador.\n"
                  f"  Directorio local: {model_path}\n"
                  f"  Error: {e}\n"
                  f"  Solución: Ejecutar setup_runpod.sh para descargar el modelo.",
                  file=sys.stderr)
            sys.exit(1)

    try:
        tok = AutoTokenizer.from_pretrained(str(model_path))
        print(f"  Tokenizador cargado desde: {model_path}")
        return tok
    except Exception as e:
        print(f"FATAL: Error cargando tokenizador desde {model_path}: {e}", file=sys.stderr)
        sys.exit(1)


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    Trunca texto a exactamente max_tokens tokens como máximo.

    Por qué esto es mejor que slices de caracteres:
      text[:400] puede dar entre 80 y 200 tokens dependiendo del idioma,
      longitud de palabras y caracteres especiales.
      Esta función garantiza exactamente max_tokens tokens.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[:max_tokens]
    # decode puede añadir espacios extra al inicio; strip() los limpia
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def count_prompt_tokens(messages: list, tokenizer) -> int:
    """
    Cuenta tokens exactamente como vLLM los verá.

    Por qué usar apply_chat_template:
      vLLM aplica internamente el chat template de LLaMA 3.1 antes de
      tokenizar. El template añade tokens especiales: <|begin_of_text|>,
      <|start_header_id|>, etc. Sin aplicar el template, los conteos
      serían más bajos que los reales y los prompts caerían fuera de VI4.
    """
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(formatted, add_special_tokens=False))


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTORES DE CASOS
# ─────────────────────────────────────────────────────────────────────────────

def build_case_a(policies_text: str, tokenizer) -> list:
    """
    Construye 30 prompts del Caso A (~256 tokens).

    Estructura:
      system: instrucción compacta + políticas de servicio (truncadas)
      user: pregunta de atención al cliente

    Estrategia de truncación:
      1. Estimar tokens disponibles para el contexto de políticas.
      2. Truncar políticas a ese presupuesto.
      3. Construir prompt y contar tokens reales.
      4. Si está fuera de rango, ajustar el presupuesto ±step tokens.

    Rango válido: [218, 294] tokens.
    """
    target = VI4_TARGETS["A"]
    prompts = []

    system_prefix = (
        "Eres el asistente virtual de TechSolutions Ecuador. "
        "Responde de forma precisa, profesional y basándote en las siguientes "
        "políticas de atención al cliente:\n\n"
    )

    for i, question in enumerate(QUESTIONS_A):
        # Buscar presupuesto correcto para políticas por búsqueda binaria
        # El overhead del template + system_prefix + question es aproximadamente
        # 50-70 tokens; empezamos con un presupuesto conservador
        best_policies_budget = 150  # tokens iniciales para políticas

        result_messages = None
        final_token_count = 0

        for attempt in range(20):  # máximo 20 iteraciones de ajuste
            policies_snippet = truncate_to_tokens(
                policies_text, tokenizer, best_policies_budget
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prefix + policies_snippet,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            n_tokens = count_prompt_tokens(messages, tokenizer)

            if target["low"] <= n_tokens <= target["high"]:
                result_messages = messages
                final_token_count = n_tokens
                break
            elif n_tokens < target["low"]:
                best_policies_budget += 20
            else:
                best_policies_budget -= 10

            if best_policies_budget < 10:
                best_policies_budget = 10
                break

        if result_messages is None:
            # Usar la última versión aunque esté fuera de rango exacto
            # (el validador final lo capturará)
            result_messages = messages
            final_token_count = n_tokens

        prompts.append({
            "case": "A",
            "prompt_id": f"A_{i+1:02d}",
            "messages": result_messages,
            "token_count": final_token_count,
            "valid": target["low"] <= final_token_count <= target["high"],
        })

    return prompts


def build_case_b(profile_text: str, conversation_histories: list, tokenizer) -> list:
    """
    Construye 30 prompts del Caso B (~1024 tokens).

    Estructura:
      system: instrucción + perfil de empresa (truncado)
      [turnos del historial de conversación]
      user: pregunta de seguimiento

    Estrategia:
      1. Cada historial tiene una cantidad fija de tokens.
      2. Calcular cuántos tokens quedan para el perfil en el system message.
      3. Truncar perfil al presupuesto restante.

    Rango válido: [870, 1178] tokens.
    """
    target = VI4_TARGETS["B"]
    prompts = []

    system_prefix = (
        "Eres un asistente conversacional de TechSolutions Ecuador. "
        "Tienes acceso al perfil de la empresa y al historial de esta conversación. "
        "Responde la consulta del usuario de forma coherente con el contexto previo.\n\n"
        "Perfil de la empresa:\n"
    )

    for i, (history, question) in enumerate(
        zip(conversation_histories, QUESTIONS_B)
    ):
        # Los mensajes de historial son los turnos previos de conversación
        # El último mensaje es la nueva pregunta del usuario
        history_messages = history  # lista de {role, content}

        # Buscar presupuesto para perfil de empresa
        best_profile_budget = 300  # tokens iniciales

        result_messages = None
        final_token_count = 0

        for attempt in range(25):
            profile_snippet = truncate_to_tokens(
                profile_text, tokenizer, best_profile_budget
            )
            messages = (
                [{"role": "system", "content": system_prefix + profile_snippet}]
                + history_messages
                + [{"role": "user", "content": question}]
            )
            n_tokens = count_prompt_tokens(messages, tokenizer)

            if target["low"] <= n_tokens <= target["high"]:
                result_messages = messages
                final_token_count = n_tokens
                break
            elif n_tokens < target["low"]:
                best_profile_budget += 30
            else:
                best_profile_budget -= 20

            if best_profile_budget < 10:
                best_profile_budget = 10
                break

        if result_messages is None:
            result_messages = messages
            final_token_count = n_tokens

        prompts.append({
            "case": "B",
            "prompt_id": f"B_{i+1:02d}",
            "messages": result_messages,
            "token_count": final_token_count,
            "valid": target["low"] <= final_token_count <= target["high"],
        })

    return prompts


def build_case_c(document_text: str, tokenizer) -> list:
    """
    Construye 30 prompts del Caso C (~4096 tokens).

    Estructura:
      system: instrucción + documento completo (truncado)
      user: pregunta de análisis

    Estrategia:
      El documento es el componente dominante (~3900 tokens).
      Truncamos el documento al presupuesto restante después de contar
      el overhead del template + system prefix + user question.

    Rango válido: [3482, 4710] tokens.
    """
    target = VI4_TARGETS["C"]
    prompts = []

    system_prefix = (
        "Eres un analista de documentos empresariales de TechSolutions Ecuador. "
        "Analiza el siguiente documento y responde la consulta del usuario "
        "basándote exclusivamente en el contenido proporcionado.\n\n"
        "Documento:\n"
    )

    for i, question in enumerate(QUESTIONS_C):
        best_doc_budget = 3700  # tokens iniciales para el documento

        result_messages = None
        final_token_count = 0

        for attempt in range(25):
            doc_snippet = truncate_to_tokens(
                document_text, tokenizer, best_doc_budget
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prefix + doc_snippet,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            n_tokens = count_prompt_tokens(messages, tokenizer)

            if target["low"] <= n_tokens <= target["high"]:
                result_messages = messages
                final_token_count = n_tokens
                break
            elif n_tokens < target["low"]:
                best_doc_budget += 50
            else:
                best_doc_budget -= 30

            if best_doc_budget < 100:
                best_doc_budget = 100
                break

        if result_messages is None:
            result_messages = messages
            final_token_count = n_tokens

        prompts.append({
            "case": "C",
            "prompt_id": f"C_{i+1:02d}",
            "messages": result_messages,
            "token_count": final_token_count,
            "valid": target["low"] <= final_token_count <= target["high"],
        })

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN Y REPORTE
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_report(prompts_by_case: dict) -> bool:
    """
    Valida que cada caso tenga exactamente 30 prompts válidos.
    Imprime estadísticas de token count por caso.
    Retorna True si todo es válido, False si algo falla.
    """
    all_ok = True
    print("\n" + "─" * 60)
    print("VALIDACIÓN DEL CORPUS")
    print("─" * 60)

    for case_label, prompts in prompts_by_case.items():
        target = VI4_TARGETS[case_label]
        counts = [p["token_count"] for p in prompts]
        valid_count = sum(1 for p in prompts if p["valid"])
        invalid_prompts = [p for p in prompts if not p["valid"]]

        mean_tokens = sum(counts) / len(counts) if counts else 0
        min_tokens = min(counts) if counts else 0
        max_tokens = max(counts) if counts else 0

        status = "✓" if valid_count == PROMPTS_PER_CASE else "✗ FALLO"
        print(
            f"  Case {case_label}: {valid_count}/{PROMPTS_PER_CASE} válidos | "
            f"mean={mean_tokens:.0f} | min={min_tokens} | max={max_tokens} | "
            f"target={target['center']} [{target['low']}-{target['high']}] {status}"
        )

        if invalid_prompts:
            all_ok = False
            for p in invalid_prompts:
                print(
                    f"    ✗ {p['prompt_id']}: {p['token_count']} tokens "
                    f"(fuera de [{target['low']}, {target['high']}])"
                )

    print("─" * 60)
    if all_ok:
        print("  Corpus completo: 90/90 prompts válidos ✓")
    else:
        print(
            "  FATAL: Algunos prompts están fuera del rango de tolerancia.\n"
            "  Posible causa: archivos de contexto muy cortos para el target.\n"
            "  Revisión: verificar tamaño de archivos en data/context/",
            file=sys.stderr,
        )
    print("─" * 60)
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="INFERA: Construir corpus de 90 prompts para el benchmark."
    )
    parser.add_argument(
        "--model-dir",
        default="/workspace/models/llama3.1-8b-instruct",
        help="Directorio del modelo LLaMA 3.1 (para cargar el tokenizador).",
    )
    parser.add_argument(
        "--context-dir",
        default="data/context",
        help="Directorio con archivos de contexto de la empresa.",
    )
    parser.add_argument(
        "--conversations",
        default="data/conversations/conversation_histories.jsonl",
        help="Archivo JSONL con 30 historiales de conversación para Caso B.",
    )
    parser.add_argument(
        "--output",
        default="data/prompts/prompt_corpus.jsonl",
        help="Ruta de salida del corpus en formato JSONL.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verificar token counts sin guardar el corpus.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("INFERA — Construcción del corpus")
    print("=" * 60)

    # ── Cargar tokenizador ────────────────────────────────────────────────────
    print("\n[1/5] Cargando tokenizador...")
    tokenizer = load_tokenizer(args.model_dir)

    # ── Cargar archivos de contexto ───────────────────────────────────────────
    print("\n[2/5] Cargando archivos de contexto...")
    context_dir = Path(args.context_dir)

    def read_context(filename: str, required: bool = True) -> str:
        path = context_dir / filename
        if not path.exists():
            if required:
                print(f"FATAL: Archivo requerido no encontrado: {path}", file=sys.stderr)
                sys.exit(1)
            return ""
        content = path.read_text(encoding="utf-8").strip()
        n_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        print(f"  {filename}: {len(content):,} chars | {n_tokens:,} tokens")
        return content

    # Archivos de contexto requeridos:
    # - company_policies.md  → Caso A (políticas de atención)
    # - company_profile.md   → Caso B (perfil de empresa)
    # - internal_faq.md      → Caso C (documento largo para análisis)
    #
    # Si tu repo usa nombres diferentes, cámbialos aquí.
    # El script fallará si alguno falta, lo cual es intencional.
    policies_text = read_context("company_policies.md", required=True)
    profile_text  = read_context("company_profile.md",  required=True)
    # Para Caso C usamos el FAQ si existe, si no usamos el contrato
    faq_path = context_dir / "internal_faq.md"
    contract_path = context_dir / "sample_contract.md"
    if faq_path.exists():
        doc_text = read_context("internal_faq.md", required=False)
    elif contract_path.exists():
        doc_text = read_context("sample_contract.md", required=False)
    else:
        print("FATAL: Se necesita internal_faq.md o sample_contract.md para Caso C.", file=sys.stderr)
        sys.exit(1)

    # Verificar que el documento C tiene suficientes tokens para el target
    doc_tokens = len(tokenizer.encode(doc_text, add_special_tokens=False))
    if doc_tokens < VI4_TARGETS["C"]["center"] * 0.8:
        print(
            f"FATAL: El documento para Caso C tiene solo {doc_tokens} tokens.\n"
            f"Se necesitan al menos {int(VI4_TARGETS['C']['center'] * 0.8)} tokens.\n"
            f"Solución: reemplazar internal_faq.md por un documento más largo.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Cargar historiales de conversación ───────────────────────────────────
    print("\n[3/5] Cargando historiales de conversación (Caso B)...")
    conv_path = Path(args.conversations)
    if not conv_path.exists():
        print(f"FATAL: Archivo de conversaciones no encontrado: {conv_path}", file=sys.stderr)
        sys.exit(1)

    conversation_histories = []
    with open(conv_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"FATAL: Error en línea {line_num} de {conv_path}: {e}", file=sys.stderr)
                sys.exit(1)
            # Formato esperado: lista de mensajes [{role, content}, ...]
            # O dict con clave "messages": [{role, content}, ...]
            if isinstance(entry, list):
                conversation_histories.append(entry)
            elif isinstance(entry, dict) and "messages" in entry:
                conversation_histories.append(entry["messages"])
            elif isinstance(entry, dict) and "turns" in entry:
                # Formato alternativo con clave "turns" en lugar de "messages"
                conversation_histories.append(entry["turns"])
            else:
                print(
                    f"FATAL: Formato inesperado en línea {line_num}.\n"
                    f"  Esperado: lista de mensajes o {{\"messages\": [...]}}",
                    file=sys.stderr,
                )
                sys.exit(1)

    if len(conversation_histories) < PROMPTS_PER_CASE:
        print(
            f"FATAL: Se necesitan {PROMPTS_PER_CASE} historiales, "
            f"encontrados: {len(conversation_histories)}",
            file=sys.stderr,
        )
        sys.exit(1)
    conversation_histories = conversation_histories[:PROMPTS_PER_CASE]
    print(f"  {len(conversation_histories)} historiales cargados.")

    # ── Construir corpus ──────────────────────────────────────────────────────
    print("\n[4/5] Construyendo prompts...")
    print("  Caso A (~256 tokens)...", end=" ", flush=True)
    prompts_a = build_case_a(policies_text, tokenizer)
    print("listo.")

    print("  Caso B (~1024 tokens)...", end=" ", flush=True)
    prompts_b = build_case_b(profile_text, conversation_histories, tokenizer)
    print("listo.")

    print("  Caso C (~4096 tokens)...", end=" ", flush=True)
    prompts_c = build_case_c(doc_text, tokenizer)
    print("listo.")

    # ── Validar ───────────────────────────────────────────────────────────────
    print("\n[5/5] Validando corpus...")
    prompts_by_case = {"A": prompts_a, "B": prompts_b, "C": prompts_c}
    corpus_valid = validate_and_report(prompts_by_case)

    if not corpus_valid:
        print("\nFATAL: Corpus no válido. No se guardará el archivo.", file=sys.stderr)
        sys.exit(1)

    if args.verify_only:
        print("\n--verify-only: corpus validado, no se guardó.")
        return

    # ── Guardar ───────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = prompts_a + prompts_b + prompts_c
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            # Guardar solo los campos que el benchmark_runner necesita
            record = {
                "case":     prompt["case"],
                "prompt_id": prompt["prompt_id"],
                "messages": prompt["messages"],
                "token_count": prompt["token_count"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ Corpus guardado: {output_path} ({len(all_prompts)} prompts)")
    print(
        f"  Distribución: "
        f"{len(prompts_a)} Case A | {len(prompts_b)} Case B | {len(prompts_c)} Case C"
    )


if __name__ == "__main__":
    main()