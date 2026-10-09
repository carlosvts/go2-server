# Divergências entre o edge (go2-tvbox) e a go2-api

Levantadas em 2026-10-09, lendo `go2-api@c880e68` (branch `dev`) e `go2-tvbox@8e324bb` (branch `sense-mode-config`). Os arquivos usados estão em `discovery/`; `scripts/build_catalog.py` repete a conferência e falha se o edge passar a usar um comando que a API não tem.

## O que precisa de ação

| # | Divergência | Onde | O que fazer |
|---|---|---|---|
| 1 | O edge considera confirmado só `ok,accepted,queued,executed` (`FALLBACK_OK_STATUSES`). O contrato deste servidor responde `enqueued` e `stopped`. Sem ajuste, **toda frase aceita toca o som de recusa**. | go2-tvbox `.env` | `FALLBACK_OK_STATUSES=enqueued,stopped` na TV Box. |
| 2 | Depois de `stand_up` ("levanta"), o `move` responde 202 e o robô não anda: a API exige `balance_stand`. Nem o edge nem a API tratam isso. | go2-api `docs/essencial/balance-stand-vs-stand-up.md` | O servidor manda `balance_stand` antes de um movimento quando não sabe se o robô já está nela (`movement.prepare` no catálogo). **Validar com o robô**, em especial sentado e deitado. O edge continua com o problema nos movimentos locais. |
| 3 | "andar para frente" é 0,3 m/s e "andar para trás" é 0,5 m/s. Andar de ré mais rápido que de frente parece invertido. | go2-tvbox `phrases.json` | Copiado como está. Revisar no edge e regenerar o catálogo. |
| 4 | O modo thin não existe na branch lida: nenhum código envia `reason="wake_word"` nem `local_*` nulos. | go2-tvbox | O servidor já aceita; só foi exercitado pelos testes e por `scripts/send_wav.py`. |
| 5 | A branch `main` da go2-api está 15 commits atrás de `dev` e não tem `GET /capabilities` nem `/safety/obstacle-avoidance`. | go2-api | Rodar o servidor contra `dev`, ou pôr `GO2S_ENSURE_OBSTACLE_AVOIDANCE=0` (não recomendado). |

## Tratadas no servidor, sem ação

| Divergência | Como ficou |
|---|---|
| O edge manda `/v1/cancel` só com `{"reason"}`, sem `edge_id`. | `edge_id` é opcional no cancel. |
| O edge manda `meta` como parte multipart com `Content-Type: application/json`. | Aceito como campo de texto ou como arquivo. |
| O edge confere e liga o desvio de obstáculo antes de cada movimento local, e avisa que os movimentos do servidor não passam por lá. | O servidor faz a mesma conferência antes de um movimento (`GO2S_ENSURE_OBSTACLE_AVOIDANCE`). A ressalva da go2-api continua: ela não comprovou que o desvio filtra o `move`. |
| No edge, uma palavra de parada em qualquer lugar da frase para o robô, menos numa frase exata de movimento. | No servidor: frase só de parada, cláusula só de parada, ou frase que termina em palavra de parada. "para frente" sozinho é rejeitado, não é parada. |
| O edge tem cooldown de 2 s para comando repetido. | O servidor não tem: a repetição chega com outro `utterance_id` e substitui a fila. |
| A TV Box desiste da resposta em 5 s (`FALLBACK_TIMEOUT_S`) e só mantém uma frase em voo. | O servidor responde assim que enfileira, sem esperar a execução. Frase com mais de `GO2S_MAX_UTTERANCE_AGE_S` (4 s) não é executada. |

## Comandos

A go2-api (`dev`) aceita 19 comandos. O edge fala 9 deles.

| Comando da go2-api | Frase no edge | No catálogo |
|---|---|---|
| `stop` | para, pare, parar, stop | parada |
| `stand_up`, `sit`, `stand_down` | levanta, senta, deita | habilitados |
| `hello`, `stretch`, `finger_heart` | cumprimentar/cumprimente, alonga, coração | habilitados |
| `damp` | desligar motores | habilitado, **só por frase exata** (o robô cai) |
| `move` | andar/virar + frente/trás/direita/esquerda | 6 regras de slots |
| `rise_sit`, `balance_stand`, `recovery_stand` | — | desabilitados |
| `wiggle_hips`, `content`, `dance1`, `dance2`, `scrape`, `pose` | — | desabilitados |
| `speed` | — | fora: exige um nível numérico |

Nenhum comando do edge falta na API, e método, rota e campos batem nos nove.

O edge chama os seis movimentos de `move` com argumentos fixos. O servidor dá um ID a cada combinação, porque a resposta tem de dizer qual foi:

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
