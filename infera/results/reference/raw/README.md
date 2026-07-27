# Mediciones del conjunto de referencia

Estos doce JSONL son las salidas directas utilizadas para calcular las cifras
del paper:

```text
2 representaciones × 2 estrategias × 3 réplicas
```

Cada archivo conserva su nombre original, incluido el identificador `v3`.
No necesitas conocer la historia de esa etiqueta para ejecutar INFERA.

Verifica la integridad:

```bash
shasum -a 256 -c SHA256SUMS
```

Todos los archivos deben indicar `OK`. No edites estos JSONL ni escribas
corridas nuevas en esta carpeta.
