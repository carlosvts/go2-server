# Divergências entre a TV Box, o catálogo e a go2-api

Levantadas em 2026-10-09, lendo `go2-api@c880e68` (branch `dev`), e revistas em 2026-10-24. Os arquivos usados estão em `discovery/`; `scripts/build_catalog.py` repete a conferência e falha se `commands.json` ou `phrases.json` usarem um comando que a API não tem.

## O que precisa de ação

| # | Divergência | Onde | O que fazer |
|---|---|---|---|
| 1 | Depois de `stand_up` ("levanta"), o `move` responde 202 e o robô não anda: a API exige `balance_stand`. A API não trata isso. | go2-api `docs/essencial/balance-stand-vs-stand-up.md` | O servidor manda `balance_stand` antes de um movimento quando não sabe se o robô já está nela (`movement.prepare` no catálogo). **Validar com o robô**, em especial sentado e deitado. |
| 2 | "andar para frente" é 0,3 m/s e "andar para trás" é 0,5 m/s. Andar de ré mais rápido que de frente parece invertido. | `discovery/phrases.json` | Revisar ali e regenerar o catálogo. |
| 3 | A branch `main` da go2-api está 15 commits atrás de `dev` e não tem `GET /capabilities` nem `/safety/obstacle-avoidance`. | go2-api | Rodar o servidor contra `dev`, ou pôr `GO2S_ENSURE_OBSTACLE_AVOIDANCE=0` (não recomendado). |

## Resolvidas

| Divergência | Como ficou |
|---|---|
| A TV Box considerava confirmado só `ok,accepted,queued,executed` (`FALLBACK_OK_STATUSES`), e toda frase aceita tocava o som de recusa. | O padrão da TV Box agora é `FALLBACK_OK_STATUSES=enqueued,stopped`. |
| O envio com `reason="wake_word"` e `local_*` nulos só tinha sido exercitado pelos testes e por `scripts/send_wav.py`. | Testado com a TV Box: 75 interações em 24/10/2026. |

## Tratadas no servidor, sem ação

| Divergência | Como ficou |
|---|---|
| A parada manual da TV Box manda `/v1/cancel` só com `{"reason": "stop"}`, sem `edge_id`. | `edge_id` é opcional no cancel. |
| A TV Box manda `meta` como parte multipart com `Content-Type: application/json`. | Aceito como campo de texto ou como arquivo. |
| A go2-api não liga o desvio de obstáculo sozinha. | O servidor confere e liga o desvio antes de um movimento (`GO2S_ENSURE_OBSTACLE_AVOIDANCE`). A ressalva da go2-api continua: ela não comprovou que o desvio filtra o `move`. |
| A TV Box desiste da resposta em 5 s (`FALLBACK_TIMEOUT_S`) e só mantém uma frase em voo. | O servidor responde assim que enfileira, sem esperar a execução. Frase com mais de `GO2S_MAX_UTTERANCE_AGE_S` (4 s) não é executada. |

## Comandos

A go2-api (`dev`) aceita 19 comandos. `phrases.json` tem frase para 9 deles.

| Comando da go2-api | Frase em `phrases.json` | No catálogo |
|---|---|---|
| `stop` | para, pare, parar, stop | parada |
| `stand_up`, `sit`, `stand_down` | levanta, senta, deita | habilitados |
| `hello`, `stretch`, `finger_heart` | cumprimentar/cumprimente, alonga, coração | habilitados |
| `damp` | desligar motores | habilitado, **só por frase exata** (o robô cai) |
| `move` | andar/virar + frente/trás/direita/esquerda | 6 regras de slots |
| `rise_sit`, `balance_stand`, `recovery_stand` | — | desabilitados |
| `wiggle_hips`, `content`, `dance1`, `dance2`, `scrape`, `pose` | — | desabilitados |
| `speed` | — | fora: exige um nível numérico |

Nenhum comando de `commands.json` falta na API, e método, rota e campos batem nos nove.

`phrases.json` dá argumentos fixos aos seis movimentos de `move`. O servidor dá um ID a cada combinação, porque a resposta tem de dizer qual foi:

| ID (escolha do servidor) | Verbo + direção | `move` |
|---|---|---|
| `move_forward` | andar + frente | `vx=0.3` |
| `move_backward` | andar + trás | `vx=-0.5` |
| `strafe_right` / `strafe_left` | andar + direita / esquerda | `vy=-0.5` / `vy=0.5` |
| `turn_right` / `turn_left` | virar + direita / esquerda | `vyaw=-0.5` / `vyaw=0.5` |

Todos com `duration_s=1.0`. "virar" + frente/trás não existe e é rejeitado.

## O que a go2-api não oferece

- **Controle exclusivo (lease):** planejado (issue #7), não implementado. Qualquer cliente comanda o robô, e o último `move` substitui o anterior. O servidor não tem o que respeitar hoje; quando o lease existir, entra em `app/go2_client.py`.
- **Sinal de conclusão:** todo comando responde `202` na hora. A fila espera o `max_duration_s` do catálogo, que é inferido.
