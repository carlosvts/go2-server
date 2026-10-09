# Riscos residuais e limites

## Se o link TV Box ↔ PC cair

O `POST /v1/cancel` e a parada falada do modo thin viajam por esse link. Se ele cair com comandos na fila, **nada avisa o servidor**: a fila continua até o fim.

O que limita o dano são só os tetos de duração:

| Limite | Padrão | Onde |
|---|---|---|
| Duração de um `move` | 1,0 s | `catalog.yaml` (vem do edge) |
| Teto do `move` na go2-api | 3,0 s | `GO2_MOVE_MAX_DURATION_S` da go2-api |
| Comandos por frase e na fila | 3 e 5 | `GO2S_MAX_CLAUSES`, `GO2S_QUEUE_MAX_COMMANDS` |
| Duração total da fila | 15 s | `GO2S_QUEUE_MAX_TOTAL_S` |
| Idade máxima da frase | 4 s | `GO2S_MAX_UTTERANCE_AGE_S` |

No pior caso com os padrões, o robô executa até 15 s de comandos sem que a TV Box consiga interromper, dos quais no máximo 3 são movimentos de 1 s. No modo edge a parada local continua funcionando, porque vai direto da TV Box à go2-api. No modo thin não há esse caminho: a parada depende do servidor.

Tenha sempre um meio de parar que não passe pela voz (o controle físico, ou um `curl -X POST <api>/commands/stop` pronto).

## Outros limites conhecidos

- **A parada não interrompe gesto.** `StopMove` para de andar. A go2-api não comprovou que ele corta um gesto ou uma postura em curso. Depois de uma parada a fila está vazia, mas o gesto que já começou pode terminar.
- **A parada espera um envio em curso.** Para garantir que o `stop` chegue à go2-api depois de um comando que já estava a caminho, ele espera esse envio terminar: no pior caso, `GO2S_GO2_API_TIMEOUT_S` (2 s) se a API travar.
- **Estado do robô é suposição.** O servidor não lê a postura do robô. Ele supõe "pronto para andar" depois de mandar `balance_stand` ou um `move`, e deixa de supor depois de postura, gesto, falha ou comando local do edge.
- **Durações inferidas.** Sem sinal de conclusão, a fila espera o `max_duration_s` do catálogo. Curto demais, o próximo comando atropela o gesto; longo demais, a fila fica lenta. Todos estão marcados `REVISAR`.
- **Desvio de obstáculo.** O servidor garante o desvio ligado antes de andar, não que o robô desvie (ressalva da própria go2-api).
- **Sem autenticação.** Quem alcança a porta 9000 comanda o robô, como na go2-api. Só na rede do laboratório.
- **Fila em memória.** Reiniciar o servidor esvazia a fila e a memória de idempotência.
- **Áudio capturado.** Com `GO2S_SAVE_UTTERANCES=1` ficam gravadas as vozes de quem falou. Avise as pessoas e não versione `captures/`.
