# CNN para escores de Gardner em blastocistos

Projeto da disciplina de Deep Learning (UTFPR IAGA): uma CNN multi-saída prevê EXP, ICM e TE a partir de imagens de blastocistos humanos.

- `gardner_cnn.ipynb`: pipeline completo
- `spec.md`: especificação
- `plan.md`: acompanhamento da execução

## Como executar

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m jupyter lab gardner_cnn.ipynb
```

Dados: Kromp et al. (2023, *Sci Data*). Os CSVs de rótulos estão versionados em `data/`; as imagens não são versionadas e devem ser colocadas em `data/images/`.
