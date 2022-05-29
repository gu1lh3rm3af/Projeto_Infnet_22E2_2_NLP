<img src="https://www.infnet.edu.br/infnet/wp-content/uploads/sites/6/2021/10/infnet_mod.svg" alt="drawing" width="500"/>

# Projeto do curso de NLP

>Curso: Processamento de linguagem natural com Python [22E2_2]

>Aluno: Guilherme Aguiar de Freitas

## Formulário de mapeamento de competências


#### Implementar técnicas de lematização

1. Qual o endereço do seu notebook (colab) executado? Use o botão de compartilhamento do colab para obter uma url.

>Link para o Repositório no GitHub:
><https://github.com/gu1lh3rm3af/Projeto_Infnet_22E2_2_NLP>

2. Em qual célula está o código que realiza o download dos pacotes necessários para tokenização e stemming usando nltk?

> Célula 5

3. Em qual célula está o código que atualiza o spacy e instala o pacote pt_core_news_lg?

> Célula 3

4. Em qual célula está o download dos dados diretamente do kaggle?

> Célula 1

5. Em qual célula está a criação do dataframe news_2016 (com examente 7943 notícias)?

> Células 7 e 8

6. Em qual célula está a função que tokeniza e realiza o stemming dos textos usando funções do nltk?

> Célula 9

7. Em qual célula está a função que realiza a lematização usando o spacy?

> Célula 13

8. Baseado nos resultados qual a diferença entre stemming e lematização, qual a diferença entre os dois procedimentos? Escolha quatro palavras para exemplificar.

> Ambas são técnicas de normalização, utilizadas para tentar extrair um significado das palavras e reduzir repetições e redundâncias nos textos analisados.
>
>O Stemming é uma técnica que extrai as raízes da palavra, fazendo um corte no início ou no final da palavra, resultando em prefixos ou sufixos comuns entre palavras flexionadas. Esse radical resultante nem sempre é uma palavra existente no vocabulário.
> Ex.: _selecionado = selec_ | _rendimento = rend_ | _falta = falt_ | _dinheiro = dinh_
>
>A Lematização é uma técnica que leva em consideração o sentido morfológico da palavra, o contexto e a classe gramatical para formar um radical comum, e este radical é uma palavra existente no vocabulário.                               
> Ex.: _selecionado = selecionar_ | _rendimento = render_ | _falta = faltar_ | _dinheiro = dinheiro_

#### Construir um modelo de reconhecimento de entidades (NER) usando Spacy.

9. Em qual célula o modelo pt_core_news_lg está sendo carregado? Todos os textos do dataframe precisam ser analisados usando os modelos carregados. Em qual célula isso foi feito?

> Célula 11

10. Indique a célula onde as entidades dos textos foram extraídas. Estamos interessados apenas nas organizações.

> Célula 16

11. Cole a figura gerada que mostra a nuvem de entidades para cada tópico obtido (no final do notebook)

<img src="https://i.ibb.co/C0VSh2p/wordcloud-ents.png" alt="drawing" width="500"/>

#### Criar modelos utilizando vetorização de textos baseado em Bag of Words

12. Quando adotamos uma estratégia frequentista para converter textos em vetores, podemos fazê-lo de diferentes maneiras. Mostramos em aula as codificações One-Hot, TF e TF-IDF. Explique a principal motivação em adotar TF-IDF frente as duas outras opções.

> As abordagens One Hot e TF dão igual importância a todas as palavras, se preocupando apenas com a frequencia da palavra. Na abordagem TF-ID a importancia das palavras é determinada através do texto, onde basicamente as palavras comumente utilizadas terão um peso menor e as palavras consideradas 'raras' terão um peso maior. É a abordagem mais popular para NLP. 

13. Indique a célula onde está a função que cria o vetor de TF-IDF para cada texto. 

> Célula 18

14. Indique a célula onde estão sendo extraídos os tópicos usando o algoritmo de LDA.

> Célula 23

15. Indique a célula onde a visualização LDAVis está criada.

> Célula 22

16. Cole a figura com a nuvem de palavras para cada um dos 9 tópicos criados.

<img src="https://i.ibb.co/dkFQsM0/wordcloud-topic.png" alt="drawing" width="500"/>

17. Escreva brevemente uma descrição para cada tópico extraído. Indique se você considera o tópico extraído semanticamente consistente ou não. 

> Tópico 1 = Documentos referentes à Estatais. Consistênte.

> Tópico 2 = Documentos referentes a ações e bolsa de valores. Consistênte.

> Tópico 3 = Documentos referentes a Receita Federal, Impostos, etc. Consistênte.

> Tópico 4 = Inconsistênte

> Tópico 5 = Documentos referentes à crédito (Empréstimos, Certão, Financiamentos). Consistênte.

> Tópico 6 = Documentos referentes à Economia. Consistênte.

> Tópico 7 = Inconsistênte

> Tópico 8 = Documentos referentes à Política. Consistênte.

> Tópico 9 = Documentos referentes à Empresas. Pouco Consistênte.

#### Criar modelos baseados em Word Embedding
18. Neste projeto, usamos TF-IDF para gerar os vetores que servem de entrada para o algoritmo de LDA. Quais seriam os passos para gerar vetores baseados na técnica de Doc2Vec?

> Primeiramente, similar a aplicação do TF_IDF, seria necessário o pré-processamento do dataset, passando as palavras para lowercase, removendo stopwords e números, e dividindo as palavras em tokens. 
Para alimentação do modelo, é necessário criar um documento taggeada, onde cada registro conterá as palavras do documento, uma tag identificadora, e a classe.
Criar o modelo com a classe Doc2Vec, utilizando a parametrização apropriada. E com o documento taggeado chamar o método build_vocab para construir o vocabulário. Aqui é importante considerar que, se a aplicação rodará com os dados fechados, é importante construir o vocabulário no conteúdo completo, porém caso seja uma aplicação viva, que receberá novos documentos constantemente, é importante rodar num dataset de treino e inferir no documentos de teste, simulando a aplicação em produção.
Por fim, com a função infer_vector, gerar os vetores do documento.

19. Em uma versão alternativa desse projeto, optamos por utilizar o algoritmo de K-Médias para gerar os clusters (tópicos). Qual das abordagens (TF-IDF ou Doc2Vec) seria mais adequada como processo de vetorização? Justifique com comentários sobre dimensionalidade e relação semântica entre documentos.

> Acredito que a abordagem com TF-IDF seria mais adequada por dois principais motivos, primeiro a dimensão do dos documentos analisados, que por se tratarem de notícias, não são tão grandes e complexos, segundo e mais importante é o fator do TF-IDF gerar vetores mais esparso, o que facilida a divisão dos clusteres.

20. Leia o artigo "Introducing our Hybrid lda2vec Algorithm" (https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=) . O algoritmo lda2vec pretende combinar o poder do word2vec com a interpretabilidade do algoritmo LDA. Em qual cenário o autor sugere que há benefícios para utilização deste novo algoritmo?

> O Autor sugere que que o lda2vec pode ser útil, numa aplicação profunda, onde você deseja fazer uma interpretação de tópicos robusta (Doc2Vec), por exemplo, porém necessita da clareza e explicabilidade do LDA.