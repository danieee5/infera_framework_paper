# Configurar una sesión o política

INFERA permite cambiar el caso de uso y los parámetros de una compactación
periódica. No convierte automáticamente cualquier mecanismo de memoria en un
experimento válido.

## Base de conocimiento

El runner espera estos archivos dentro de `infera/kb/`:

- `vigia_kb.md`;
- `permisos_medicos.csv`;
- `inventario_uniformes.csv`.

Puedes reemplazar su contenido manteniendo los nombres, o adaptar
`infera_kb.py`. Utiliza datos ficticios, anonimizados o autorizados.

El tamaño debe evaluarse en tokens. En cada petición deben caber sistema,
historial activo, pregunta y salida reservada.

## Tareas

Cada tarea necesita un `id` único, `type`, `prompt`, `depends_on` y `verify`.

Las reglas implementadas son:

- `contains_all`: exige todos los términos.
- `contains_any`: acepta variantes declaradas.
- `forbidden`: penaliza términos no permitidos.
- `required_fields`: comprueba secciones de una salida estructurada.
- `rota`: valida cobertura y restricciones de turnos.

El campo `judge` de configuraciones antiguas no activa una evaluación LLM. La
medición actual usa `score_task`.

## Decodificación

- Temperatura 0 reduce muestreo aleatorio.
- Una semilla fija mejora repetibilidad local, pero no garantiza igualdad
  entre motores o hardware.
- `max_tokens` reserva la salida máxima de las tareas y resúmenes.

## Presupuesto de contexto

`MAX_MODEL_LEN=8192` significa que una petición completa debe caber en ese
presupuesto. No limita la cantidad total de consultas durante la vida del
servidor.

`THRESH` se compara con los tokens del prompt informados por vLLM. Debe
elegirse antes de medir. Un valor menor puede generar más resúmenes; uno mayor
conserva más historial por petición. La conveniencia energética debe medirse.

## Validación

Desde `infera/`:

```bash
python config/validate_configuration.py \
  --tasks config/mi_sesion.json \
  --kb-dir kb \
  --max-model-len 8192 \
  --threshold 4500 \
  --tokenizer /ruta/al/tokenizador
```

Después ejecuta un humo y comprueba estado, NVML, energía positiva,
truncamiento, cantidad de compactaciones y reglas programáticas.

Consulta también
[`../infera/config/README.md`](../infera/config/README.md).
