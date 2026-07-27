# Configurar una medición

Esta carpeta contiene una plantilla para una sesión propia y, separadamente,
la configuración exacta del conjunto de referencia.

## Archivos para una corrida nueva

- `experiment.env.example`: variables de modelos, política y ejecución.
- `session_tasks.example.json`: tres tareas ficticias para probar el flujo.
- `validate_configuration.py`: valida estructura, dependencias y presupuesto.

Prepara copias editables:

```bash
cp config/experiment.env.example config/experiment.env
cp config/session_tasks.example.json config/mi_sesion.json
```

En `config/experiment.env`, cambia `SESSION` para que apunte a
`config/mi_sesion.json`.

## Diseñar las tareas

Cada tarea necesita:

- `id` único;
- `type`;
- `prompt`;
- `depends_on`, aunque sea una lista vacía;
- `verify`, con reglas que correspondan a la respuesta esperada.

Las reglas disponibles son:

- `contains_all`: exige todos los términos declarados.
- `contains_any`: acepta al menos una variante de cada grupo.
- `forbidden`: penaliza términos o entidades no permitidas.
- `required_fields`: comprueba campos de una salida estructurada.
- `rota`: valida cobertura y restricciones de turnos.

Define respuestas comprobables y adapta las reglas cuando reemplaces la base.
El campo `judge` que pudiera existir en una configuración antigua no activa un
juez LLM: el runner usa la rúbrica programática.

## Elegir la longitud de la sesión

El ejemplo de tres tareas sirve solo para una prueba de humo y normalmente no
activa la compactación. Para medir una política debes:

1. crear una secuencia suficientemente larga;
2. definir `THRESH` antes de ejecutar;
3. contar con el tokenizador real;
4. reservar espacio para la salida;
5. declarar `EXPECTED_TASKS` y `EXPECTED_COMPACTIONS`;
6. mantener la misma secuencia para ambos brazos.

No existe un número universal de documentos, palabras o tareas. El límite se
controla en tokens:

```text
mensaje de sistema + historial activo + pregunta + salida reservada
```

`MAX_MODEL_LEN` limita cada petición, no todos los tokens procesados durante
la vida del servidor.

## Decodificación

- Temperatura 0 reduce el muestreo aleatorio.
- Una semilla fija mejora repetibilidad local, pero no garantiza igualdad
  entre versiones, motores o GPU.
- `max_tokens` reserva el máximo de salida de cada tarea y de cada resumen.

## Validar antes de usar GPU

```bash
python config/validate_configuration.py \
  --tasks config/mi_sesion.json \
  --kb-dir kb \
  --max-model-len 8192 \
  --threshold 4500 \
  --tokenizer /ruta/al/modelo-o-tokenizador
```

El conteo inicial no puede predecir exactamente cuánto crecerán las
respuestas. Ejecuta después una réplica de humo y revisa los `prompt_tokens`
informados por vLLM.

## Configuración del conjunto de referencia

`reference/session_tasks.json` conserva la secuencia exacta utilizada para las
mediciones publicadas. Su descripción interna registra la motivación
histórica del diseño y no debe interpretarse como un umbral óptimo.

`reference/experiment.env.example` documenta los parámetros necesarios para
repetir esa configuración. No es el archivo que debe usar una persona para
crear su propio caso.
