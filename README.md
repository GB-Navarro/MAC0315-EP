## 1. Objetivo do projeto
O projeto tem como objetivo principal implementar e testar uma versão do método simplex revisado para a solução de problemas de programação linear. Este método é amplamente reconhecido como uma ferramenta fundamental em otimização, sendo utilizado para encontrar soluções ótimas em problemas que envolvem múltiplas variáveis e restrições lineares. A proposta busca desenvolver uma abordagem que garanta precisão nos resultados, eficiência computacional e flexibilidade no tratamento de diferentes tipos de problemas.

A proposta inclui a utilização da linguagem Python, com suporte da biblioteca numpy para operações matriciais. 

Entre os principais aspectos que serão abordados estão:

1. Determinar uma solução básica factível inicial;

2. Implementar atualizações eficientes das bases, com o uso de decomposições matriciais, como a decomposição LU;

3. Validar a implementação por meio de exemplos de programação linear, comparando os resultados obtidos com soluções geradas por pacotes que já são amplamente utilizados, como o scipy.optimize, por exemplo.
   
## 2. Formato dos arquivos de entrada

*(Ver o arquivo 'entry_description.txt' e os arquivos nas pastas 'Lista 1' e 'Lista 2')*

O formato dos arquivos de entrada foi projetado para descrever problemas de programação linear de forma clara e estruturada, permitindo sua leitura pelo programa 'simplex.py'.

A primeira linha identifica o tipo de problema, especificando se a função objetivo será maximizada ou minimizada, utilizando as palavras-chave max ou min. Essa informação define o objetivo do algoritmo ao processar o problema.

Em seguida, os coeficientes da função objetivo são listados na linha denominada "Custos das variáveis de decisão". Estes coeficientes representam os pesos das variáveis na função objetivo e devem ser especificados em sequência, separados por espaços.

As restrições lineares são descritas sob o cabeçalho "Restrições". Cada linha corresponde a uma restrição, formada pelos coeficientes das variáveis de decisão, seguidos pelo operador relacional (que pode ser <=, >= ou ==), e pelo valor do lado direito da equação.

Os limites das variáveis de decisão são definidos na seção final, intitulada "Limites das variáveis de decisão". Para cada variável, é indicado se ela deve ser não negativa (xi >= 0) ou se possui limites inferiores e superiores (j <= xi <= k), onde j e k são números reais que delimitam o intervalo permitido para a variável.

A sintaxe do formato requer atenção a detalhes específicos:

1. Não devem haver espaços extras desnecessários, pois eles não são tratados pelo programa.
2. Problemas fora desse formato não serão interpretados corretamente.

## 3. Executando o programa 'simplex.py'
O arquivo 'simplex.py' requer como entrada o caminho de um arquivo .txt contendo a definição do problema de programação linear no formato especificado anteriormente. Opcionalmente, é possível ativar um modo "verboso" que exibe detalhes intermediários da execução. A sintaxe geral para a execução é a seguinte:

        python3 simplex.py [caminho do .txt que contém o problema] [true (opcional)]

**Argumento obrigatório:** o primeiro argumento é o caminho para o arquivo .txt com a definição do problema.

**Argumento opcional:** o segundo argumento, true, habilita o modo verboso, exibindo o passo a passo do algoritmo.

**Exemplos de uso:**

1. Para executar sem informações detalhadas *(nesse caso, apenas o resultado final será exibido)*:
   
        python3 simplex.py Lista\ 2/l2_ex4.txt

2. Para executar com o modo verboso ativado *(aqui, o programa exibirá informações detalhadas sobre cada iteração do método simplex, incluindo mudanças na base e cálculos intermediários.)*: 

        python3 simplex.py Lista\ 2/l2_ex4.txt true

## 4. Executando o programa 'simplex_scipy.py'
O arquivo 'simplex_scipy.py' utiliza o módulo scipy.optimize para resolver os problemas de programação linear das pastas 'Lista 1' e 'Lista 2'. Sua execução é mais simples do que a execução de 'simplex.py', veja:

        python3 simplex_scipy.py

Para esse script, os problemas são definidos internamente, mais precisamente na função 'define_problems'.

*OBS: O script 'simplex_scipy.py' foi desenvolvido com o único objetivo de servir como referência para comparação dos resultados obtidos pela implementação personalizada do método simplex revisado no arquivo 'simplex.py'.*

## 5. Breve explicação da implementação
Abaixo estão os principais componentes da solução implementada:

**Processamento de Entrada**

O script inicia com uma função de processamento de entrada ‘process_entry’ que transforma um problema de otimização linear definido textualmente em variáveis manipuláveis. São extraídos:

-  O tipo do problema (maximização ou minimização).

- O vetor de coeficientes da função objetivo.

- A matriz de restrições e o vetor de recursos.

- As relações entre coeficientes e recursos (<=, >= ou =).

**Ajuste do Problema para a Forma Padrão**

A função de pré-processamento ‘preprocess_problem’ tem como objetivo adaptar o problema original à forma padrão, convertendo desigualdades em igualdades, o que é feito por meio da introdução de variáveis auxiliares. Essa etapa também pode incluir a adição de novas restrições, dependendo dos limites das variáveis de decisão, e realizar a inversão de eventuais restrições que envolvam recursos negativos. Além disso, a função é responsável por identificar se será necessário aplicar o método das duas fases.

**Resolução utilizando Decomposição LU**

Uma abordagem personalizada de decomposição LU é implementada para resolver o sistema linear $Bx_{b} = b$ no método simplex:

1. A matriz de restrições é decomposta em componentes triangulares inferiores (L), superiores (U) e de permutação (P).

2. Sistemas lineares intermediários são solucionados eficientemente com substituição direta e retroativa.

**Algoritmo Simplex Revisado**

Além da decomposição LU para resolver $Bx_{b} = b$, o algoritmo principal ‘simplex_revised’ realiza:

- Cálculo iterativo de custos reduzidos para avaliar otimalidade.

- Determinação da variável que entra na base por meio do método de maior (ou menor) custo reduzido.

- Determinação da variável que sai da base com base na regra de razão mínima, verificando limitações do conjunto viável.

- Atualizações sucessivas das bases até atingir a solução ótima ou detectar problemas ilimitados.

**Método das Duas Fases**

Para problemas que exigem viabilidade inicial, é utilizado o método das duas fases. Nesse processo, a Fase 1 do algoritmo minimiza a soma das variáveis artificiais, assegurando que o problema seja convertido em uma forma viável. Em seguida, a Fase 2 utiliza essa solução inicial para resolver o problema original.

**Integração e Visualização**

A função principal ‘main’ integra todas as etapas:

- Carrega os dados do problema.

- Aplica o pré-processamento.

- Exibe o problema original e ajustado.

- Resolve o problema utilizando o método adequado.

- Apresenta a solução ótima e o tempo de execução.

## 6. Eventuais limitações da implementação
Embora os resultados dos testes tenham sido bem-sucedidos, podem surgir problemas inesperados. Dentre eles, destacam-se:

1. **Instabilidade Numérica:** Apesar do uso de pivotamento parcial na decomposição LU, matrizes mal condicionadas podem gerar erros numéricos, comprometendo a precisão das soluções.

2. **Problemas Ilimitados e Inviáveis:** Embora mensagens de erro sejam emitidas para essas condições, o algoritmo não implementa diagnósticos detalhados, o que pode vir a ser um problema.
   
3. **Falta de Verificação de Entrada:** A validação dos dados de entrada é limitada, podendo resultar em erros em casos de formatos inesperados ou inconsistências nos arquivos.

4. **Escopo Limitado de Testes:** O programa foi testado em um número pequeno de problemas (apenas os problemas na pasta "Lista 1" e "Lista 2"). Portanto, é possível que determinadas combinações de restrições ou limites das variáveis de decisão causem comportamentos inesperados ou erros que não foram detectados durante o desenvolvimento.
   
## 7. Resultados obtidos
Nesta seção, apresentamos os resultados obtidos pela implementação customizada do método simplex revisado e comparamos esses resultados com aqueles gerados utilizando o módulo scipy.optimize.

**Comparativo das Soluções**

Para todos os problemas resolvidos *(Lista 1 - Exercícios 2.1 a 2.5 e Lista 2 - Exercícios 3 e 4)*, a solução ótima encontrada pela implementação customizada apresentou os mesmos valores ótimos de Z quando comparada à solução do módulo scipy.optimize. Essa equivalência foi verificada para os casos em que as soluções correspondem a um ponto específico, bem como para casos onde a solução está em um segmento de reta.

No entanto, para problemas como o Exercício 2.4 da Lista 1 e o Exercício 4 da Lista 2, onde a solução corresponde a uma reta, os valores individuais das variáveis $x_{1}, x_{2}$ e  $x_{3}$ apresentaram diferenças entre as duas implementações. Essa diferença é esperada e não compromete a validação, visto que ambas respeitam as condições de ótimo e pertencem à região viável.

**Comparativo de Desempenho (Tempo de Execução)**

Os tempos de execução foram avaliados em todas as execuções e mostram uma diferença significativa entre as duas abordagens. A seguir, destacamos os tempos médios de resolução para cada problema:

| Problema         | Tempo Médio (Custom) | Tempo Médio (Scipy) |
|------------------|-----------------------|----------------------|
| Lista 1 - 2.1   | 0,00182 s            | 0,01519 s           |
| Lista 1 - 2.2   | 0,00275 s            | 0,00297 s           |
| Lista 1 - 2.3   | 0,00481 s            | 0,00253 s           |
| Lista 1 - 2.4   | 0,00230 s            | 0,00209 s           |
| Lista 1 - 2.5   | 0,00368 s            | 0,00254 s           |
| Lista 2 - 3     | 0,00912 s            | 0,00201 s           |
| Lista 2 - 4     | 0,00452 s            | 0,00202 s           |

Observa-se que, em termos de eficiência, o módulo scipy.optimize apresentou desempenho superior na maioria dos casos, especialmente em problemas como o Exercício 3 da Lista 2 *(problema difícil)*, onde o tempo de execução médio do módulo scipy.optimize foi mais de quatro vezes menor que o da implementação customizada. Por outro lado, em problemas como o Exercício 2.1 da Lista 1 *(problema fácil)*, a implementação customizada demonstrou tempos comparativamente menores.

**Conclusão**

Em suma, os resultados obtidos evidenciam a corretude, eficiência e robustez da implementação customizada do método simplex revisado. Apesar de ser esperado que o módulo scipy.optimize apresente maior velocidade de execução devido as suas otimizações internas, a implementação desenvolvida mostrou-se funcional e confiável nos testes realizados, sendo capaz de resolver problemas de programação linear com resultados comparáveis aos obtidos pelo módulo em questão.
