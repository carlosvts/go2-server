# go2-server

Servidor de inferência e despacho dos comandos de voz do Go2 (NEURON/UFLA). Roda no mesmo PC da [go2-api](https://github.com/carlosvts/go2-api). A TV Box ([go2-tvbox](https://github.com/carlosvts/go2-tvbox)) só detecta a wake word e manda o áudio da frase; este servidor transcreve, interpreta, mantém a fila de execução e chama a go2-api.

```
TV Box ──áudio──▶ go2-server: transcrição → interpretação → fila ──HTTP──▶ go2-api ──▶ robô
        ◀─status─┘
```

A TV Box não executa nada do que o servidor responde: o `status` só serve para o retorno sonoro.

A interpretação **não usa LLM**. Ela só seleciona comandos de um catálogo fixo (`catalog.yaml`), por correspondência exata, por slots de verbo e direção, e por vizinho mais próximo em embeddings. Não existe caminho que gere um comando fora do catálogo.

## Estado

- **Validado:** 161 testes (`uv run pytest`), com a go2-api e a transcrição falsas; quatro deles rodam o conjunto rotulado no modelo de embeddings real. Ponta a ponta com a TV Box: 75 interações em 24/10/2026.
- **Não validado:** veja [O que falta validar](#o-que-falta-validar).

## Como rodar

### 1. Pré-requisitos

- Linux, no mesmo PC da go2-api.
- [uv](https://docs.astral.sh/uv/) instalado.
- A go2-api (branch `dev`) no ar. Confira:

  ```bash
  curl http://127.0.0.1:8000/capabilities
  ```

  Tem de devolver a lista de comandos. A branch `main` da go2-api não tem essa rota nem a do desvio de obstáculo.
- Para o Whisper `medium` (padrão), uma GPU NVIDIA com driver instalado. Sem GPU, veja o passo 3.

### 2. Instale

```bash
git clone git@github.com:carlosvts/go2-server.git
cd go2-server
uv sync
```

### 3. Configure

```bash
cp .env.example .env
```

O `.env.example` já serve como está quando a go2-api roda em `127.0.0.1:8000`. Se não, ajuste `GO2S_GO2_API_URL`. Sem GPU, descomente e troque o modelo:

```
GO2S_WHISPER_MODEL=small
```

O `medium` na CPU pode demorar mais que os 5 s que a TV Box espera pela resposta.

### 4. Suba o servidor

```bash
uv run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 9000
```

Na primeira subida o servidor baixa o modelo do Whisper (`medium`, ~1,5 GB) e o de embeddings (~470 MB), então demora. Espere no log:

- `Whisper medium em cuda (float16)` (ou `cpu`, se você escolheu a CPU);
- `go2-server pronto`, quando já aceita frases.

### 5. Confira

```bash
curl http://127.0.0.1:9000/health
```

A resposta diz se a go2-api responde e quantos comandos há na fila.

### 6. Teste sem a TV Box

```bash
uv run python scripts/send_wav.py --say "hey jarvis senta" --reason wake_word
uv run python scripts/send_wav.py frase.wav --reason wake_word
uv run python scripts/send_wav.py --cancel stop
```

A primeira linha faz o robô sentar: teste com ele num lugar seguro. `--say` sintetiza a frase (precisa de `sox` e de `espeak-ng`, ou de uma voz do Piper: veja `scripts/tts.py`). A resposta impressa traz `status`, `transcript` e `commands`.

### 7. Aponte a TV Box para o servidor

No `.env` do [go2-tvbox](https://github.com/carlosvts/go2-tvbox):

```
SERVER_URL=http://<IP_DO_PC>:9000
```

O padrão da TV Box (`FALLBACK_OK_STATUSES=enqueued,stopped`) já bate com os status deste servidor. Diga "hey jarvis", espere o beep e diga o comando. Tenha a parada manual da TV Box (`uv run python -m sense.stop`) ou o controle físico à mão.

### GPU

`GO2S_WHISPER_DEVICE=auto` (padrão) usa CUDA se houver. O processo precisa enxergar as bibliotecas cuBLAS e cuDNN 9 para CUDA 12. Sem elas no sistema:

```bash
uv sync --extra gpu
export LD_LIBRARY_PATH=$(uv run python -c 'import nvidia.cublas, nvidia.cudnn; print(nvidia.cublas.__path__[0] + "/lib:" + nvidia.cudnn.__path__[0] + "/lib")')
```

Rode o `export` no mesmo terminal antes do passo 4. Se o log mostrar `Whisper não subiu em cuda (...); tentando cpu`, a GPU falhou e o servidor caiu para a CPU: corrija a GPU ou use `GO2S_WHISPER_MODEL=small` ou `base`.

### Com Docker, ao lado da go2-api

No lugar dos passos 2 a 4:

```bash
cp .env.example .env
docker compose --profile server up -d --build
docker compose logs -f go2-server
```

O container usa a rede do host, como o da go2-api, e tem teto de 4 CPUs para não tirar processador do reenvio do `Move`. Para o container usar a GPU, instale o NVIDIA Container Toolkit e descomente `gpus: all` no `docker-compose.yml`. Como serviço systemd, sem Docker: instruções em `deploy/go2-server.service`.

## Contrato

### `POST /v1/utterance` (multipart)

| Campo | Conteúdo |
|---|---|
| `audio` | WAV mono 16 kHz PCM16 |
| `meta` | JSON: `utterance_id`, `edge_id` (quem enviou, ex. a TV Box), `reason`, `local_hypothesis`, `local_confidence` |

A TV Box manda `reason="wake_word"` e `local_*` nulos. Outro `reason` é aceito e logado. Só responde `422` para áudio ausente ou em outro formato, `meta` ilegível e `utterance_id` ausente.

Resposta `200`:

```json
{"utterance_id": "…", "status": "enqueued", "transcript": "Senta.",
 "commands": [{"id": "sit"}], "confidence": 1.0, "reject_reason": null}
```

| `status` | Significa |
|---|---|
| `enqueued` | a frase virou comandos e entrou na fila |
| `stopped` | era parada: fila limpa e `stop` entregue à go2-api |
| `ignored` | só a wake word, sem comando |
| `rejected` | nada foi executado; o motivo vem em `reject_reason` |

`confidence` é o menor score entre as cláusulas (1,0 quando a correspondência é exata). Os motivos de rejeição mais comuns: `negation`, `low_similarity`, `ambiguous`, `movement_missing_verb`, `movement_missing_direction`, `command_disabled`, `too_many_clauses`, `stale`, `no_speech`, `low_confidence`, `hallucination`, `stop_not_delivered`.

A repetição de um `utterance_id` devolve a mesma resposta e não executa nada de novo (memória de 10 min).

### `POST /v1/cancel`

`{"reason": "stop" | "local_command", "edge_id": "…"}` (`edge_id` opcional). A parada manual da TV Box manda `{"reason": "stop"}`. Limpa a fila e interrompe o comando em andamento. Com `stop`, também chama `POST /commands/stop` na go2-api; repetir é inofensivo. Responde `{"cleared": n, "stop_sent": true|false|null}`.

### `GET /health` e `GET /v1/stats`

`/health` informa a fila pendente, o último erro de despacho e se a go2-api responde. `/v1/stats` traz, por `reason`, a contagem por `status`, a latência média e p95, e a latência das paradas à parte.

## Como uma frase é interpretada

1. **Transcrição** livre em pt-BR (faster-whisper, com VAD e as defesas contra alucinação descritas em `app/transcribe/whisper.py`).
2. **Wake word:** removida do início quando presente ("hey jarvis" e as grafias em `GO2S_WAKE_WORDS`, com tolerância a erro pequeno). Se não sobra nada, `ignored`.
3. **Parada:** a frase é só parada ("para", "pare", "parar", "stop") ⇒ `stopped`. Esse caminho não usa embedding e vale mesmo para transcrição de baixa confiança: parar por engano é seguro. Também param uma cláusula só de parada ("senta e para"), uma frase que termina em palavra de parada ("pode parar") e o que se parece com parada por embedding ("fica parado"), este com limiar próprio e mais frouxo (`GO2S_TAU_STOP_ACCEPT`).
4. **Negação** ("não", "nunca", "sem") em qualquer parte ⇒ a frase inteira é rejeitada.
5. **Cláusulas:** a frase é cortada em "e", "depois", "então", "em seguida" e nas pausas, até `GO2S_MAX_CLAUSES`.
6. **Cada cláusula**, nesta ordem:
   - frase exata do catálogo (sem acento e pontuação);
   - **movimento por slots:** um verbo (`andar`, `virar` e sinônimos) e uma direção (`frente`, `trás`, `direita`, `esquerda`), cada um resolvido sozinho. "para/pra/a" é ignorado. Os dois são obrigatórios, e palavra sobrando rejeita a cláusula. A frase inteira nunca é comparada por embedding, porque "andar para direita" e "virar para direita" ficam quase idênticos;
   - vizinho mais próximo por embeddings nos exemplos do catálogo, aceito só com `cosseno ≥ τ_accept` **e** `top1 − top2 ≥ τ_margin`.
7. **Uma cláusula rejeitada rejeita a frase inteira.** Não há execução parcial.

`damp` ("desligar motores") tira a força dos motores e o robô cai: só é aceito por frase exata, nunca por embedding.

## Catálogo

`catalog.yaml` é **gerado**, não edite à mão:

```bash
uv run python scripts/build_catalog.py                       # regenera com o que há em discovery/
uv run python scripts/build_catalog.py \
    --capabilities http://127.0.0.1:8000/capabilities        # e antes relê a go2-api
```

| Arquivo em `discovery/` | Origem |
|---|---|
| `capabilities.json` | `GET /capabilities` da go2-api |
| `commands.json`, `phrases.json` | **escritos à mão: as frases que o usuário fala** |
| `inferred.yaml` | **escrito à mão: é o que você revisa** |

Para mudar as frases, edite `discovery/phrases.json` (frases que o usuário fala) e `discovery/inferred.yaml` (sinônimos, exemplos extras, durações) e rode `uv run python scripts/build_catalog.py`, sem argumentos.

Tudo o que não veio da go2-api nem de `phrases.json` está em `inferred.yaml` e sai no catálogo marcado `INFERIDO` ou `INFERIDO, REVISAR`: as durações de espera de cada comando, os sinônimos, os IDs dos movimentos, a postura de preparo e a lista de comandos desabilitados. O relatório da reconciliação está em [docs/divergencias.md](docs/divergencias.md).

## Fila

- **Um consumidor.** Para cada comando, chama a go2-api e espera o `max_duration_s` do catálogo. A API não avisa o fim de um gesto, então esse tempo é o único "fim" conhecido.
- **A fila é global.** O robô é um só, então frases de `edge_id`s diferentes disputam a mesma fila, e por padrão a frase nova de um cliente substitui a de outro. `GO2S_CROSS_EDGE_POLICY=reject` recusa a frase de outro cliente enquanto houver fila; parada e cancel valem sempre, de qualquer cliente.
- **Frase nova substitui a fila** e interrompe a espera do comando em andamento (`GO2S_REPLACE_QUEUE=0` para enfileirar no fim).
- **Falha na go2-api** limpa a fila: os comandos seguintes não saem.
- **Antes de um movimento** o servidor confere e liga o desvio de obstáculo, e manda `balance_stand` se não sabe se o robô já está pronto para andar.
- **Sem controle exclusivo.** A go2-api não tem lease (planejado na issue #7 dela), então não há o que respeitar hoje.

A corrida entre cancelamento e consumidor é fechada por um contador de geração e um lock em torno de "conferir a geração e enviar" (explicação no topo de `app/dispatch.py`). Um comando que já estava a caminho chega à go2-api antes do `stop`, nunca depois.

Os limites e o que acontece se o link com a TV Box cair estão em [docs/riscos.md](docs/riscos.md).

## Calibração

```bash
uv run python scripts/calibrate.py                   # precisão e revocação por limiar
uv run python scripts/calibrate.py --show 0.95,0.10  # lista os erros num limiar
uv run python scripts/bench_stt.py                   # latência, RAM e erro dos motores
```

`eval/phrases.yaml` é o conjunto rotulado (159 frases escritas à mão: positivas, compostas, negação, parada, fora do catálogo, parecidas entre si e com a wake word). A prioridade é precisão: um falso positivo move um robô físico.

Resultado no conjunto atual, que definiu os padrões:

| Configuração | Precisão | Revocação | Falsos positivos |
|---|---|---|---|
| Só lexical e slots, sem embeddings | 1,000 | 0,736 | 0 |
| MiniLM-L12, `τ_accept=0.95`, `τ_margin=0.10` (**padrão**) | 1,000 | 0,792 | 0 |
| MiniLM-L12, `τ_accept=0.90`, `τ_margin=0.10` | 0,983 | 0,819 | 1 |
| MiniLM-L12, `τ_accept=0.80`, `τ_margin=0.10` | 0,953 | 0,847 | 3 |
| multilingual-e5-small, melhor ponto sem falso positivo | 1,000 | 0,736 | 0 |

Duas conclusões: quase todo o acerto vem do caminho lexical e dos slots, e o `multilingual-e5-small` não acrescenta nada sem falso positivo (os cossenos dele ficam todos altos e a margem não separa). O jeito mais barato de subir a revocação é acrescentar exemplos em `discovery/inferred.yaml`, não baixar o limiar.

Para montar um conjunto de fala real, ligue `GO2S_SAVE_UTTERANCES=1`: cada frase recebida fica em `captures/<data>/<utterance_id>.wav` com um `.json` ao lado (transcrição antes e depois da wake word, comandos, scores, tempos). As transcrições rejeitadas que as pessoas repetem são candidatas a exemplos em `discovery/inferred.yaml`.

## Observabilidade

Uma linha JSON por evento. A de cada frase (`"event": "utterance"`) traz `utterance_id`, `edge_id`, `reason`, `transcript_raw` e `transcript` (antes e depois da wake word), `commands`, `clauses` (método, cosseno e margem de cada cláusula), `reject_reason`, os scores do motor de transcrição e os tempos `stt_ms`, `interpret_ms`, `dispatch_ms` e `total_ms`. Cada comando enviado à go2-api gera um `"event": "dispatch"` com o mesmo `utterance_id`. A cada 60 s sai o resumo por `reason`.

## Estrutura

```
app/
  main.py            app factory e rotas
  service.py         o caminho de uma frase, idempotência, log e captura
  dispatch.py        a fila e o consumidor
  go2_client.py      cliente HTTP da go2-api
  catalog.py         leitura e validação do catalog.yaml
  wake_word.py       remoção da wake word
  interpret/         normalização, interpretador, embeddings
  transcribe/        faster-whisper, Vosk e a faixa reservada para áudio curto
  metrics.py, logging_setup.py, audio.py, config.py
catalog.yaml         gerado
discovery/           capabilities.json da go2-api, frases e inferred.yaml
eval/phrases.yaml    conjunto rotulado
scripts/             build_catalog, calibrate, bench_stt, send_wav, tts
deploy/              Dockerfile e serviço systemd
docs/                divergencias.md, riscos.md
tests/
```

## O que falta validar

1. **Fala real.** A precisão e a revocação foram medidas em texto escrito à mão. Com microfone, sotaque e ruído, a transcrição erra de formas que o conjunto não cobre.
2. **Latência e memória no PC de destino, e o Whisper na GPU.** O desenvolvimento foi feito em outra máquina, sem placa. Rode `scripts/bench_stt.py` no PC da go2-api para confirmar que o `medium` sobe em `cuda` e conferir a latência da parada falada.
3. **Vosk com modelo grande.** `vosk-model-pt-fb-v0.1.1` (1,6 GB) não foi medido: o download não completou. O script já o compara se a pasta estiver em `models/`.
4. **go2-api real e o robô.** Nenhuma chamada saiu para a API de verdade. Em especial: `balance_stand` antes de andar (sentado e deitado), as durações de cada gesto, e se o `stop` interrompe um gesto.
5. **Grafias da wake word.** A lista em `GO2S_WAKE_WORDS` é um palpite. Ajuste com o que o Whisper escrever de verdade (campo `transcript_raw` do log).
6. **Disputa de CPU com a go2-api.** O teto de 4 CPUs é um chute; confira se o robô anda liso enquanto o servidor transcreve.
7. **A imagem Docker** não foi construída nem executada.
