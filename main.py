#!/usr/bin/env python

debug = True
if debug: print('debug = True')
computaGenero = True
if debug: print('computaGenero = ' + str(computaGenero))
computaCursos = True
if debug: print('computaCursos = ' + str(computaCursos))
computaRecorrencia = True # Também vale para retenção e evasão
if debug: print('computaRecorrencia = ' + str(computaRecorrencia))
entrada = True
if debug: print('entrada = ' + str(entrada))
incremental = False # Só tem valor caso entrada tenha valor True. Faz com que o que já estiver na saída seja mantido
if debug: print('incremental = ' + str(incremental))
graficos = True
if debug: print('graficos = ' + str(graficos))
if debug: print('')

import json
import pandas as pd
import os
from pathlib import Path
import re
import errno
from enum import Enum
from collections import namedtuple
import numpy as np
if computaGenero:
    from genderize import Genderize
    genderize = Genderize()

if graficos:
    import matplotlib.pyplot as plt
    #plt.close('all')

if computaRecorrencia:
    from unidecode import unidecode
    from operator import itemgetter
    from similarity.jarowinkler import JaroWinkler
    jw = JaroWinkler()

if graficos:
    import itertools
    import calendar

# Colunas que não fazem parte da entrada devem ter o valor de Expressão em branco
# Colunas com valor de Expressão tem seus nomes substituídos pela Descrição
Coluna = namedtuple('Coluna', ['Descrição', 'Expressão'])
class Colunas(Enum):
    @property
    def Descrição(self):
        '''Nome da coluna.'''
        return self.value[0].Descrição

    @property
    def Expressão(self):
        '''Regex da coluna.'''
        return self.value[0].Expressão

    # Cuidado para não esquecer da vírgula no final da cada linha
    Nome = Coluna('Nome', r'NOME'),
    RG = Coluna('RG', r'Documento de Identidade|^R\.?G\.?$'),
    CPF = Coluna('CPF', r'CPF'),
    Curso = Coluna('Curso', r'CURSO'),
    ID = Coluna('ID', None),
    Ação = Coluna('Ação', None),
    Evasão = Coluna('Evasão', None),
    Evasora = Coluna('Evasora', None),
    Gênero = Coluna('Gênero', None),
    Porcentagem = Coluna('Porcentagem', None),
    Retenção = Coluna('Retenção', None),
    Retentora = Coluna('Retentora', None),
    Quantidade = Coluna('Quantidade', None),
    Válidos = Coluna('Qtde. voluntários válidos', None),

try:
    with open("../Saida/generos.json") as json_file:
        generos = json.load(json_file)

    if debug: print('Lendo Saida/generos.json')
except FileNotFoundError:
    if debug: print('Saida/generos.json não encontrado')
    # Nomes podem ser adicionados aqui (ou no arquivo Saida/generos.json) caso não seja encontrado pela Genderize
    generos = {
        'ALDREI': 'm',
        'EDIPO': 'm',
        'FABRICIO': 'm',
        'HYTALO': 'm',
        'JOLINDO': 'm',
        'KAWE': 'm',
        'MASSARU': 'm',
        'OTAVIO': 'm',
        'VINICIUS': 'm',
        'CARINE': 'f',
        'CASSIA': 'f',
        'FLAVIA': 'f',
        'FRANCYELE': 'f',
        'GABRIELLA': 'f',
        'HELOISA': 'f',
        'IHANNA': 'f',
        'JENYFFER': 'f',
        'JESSICA': 'f',
        'JULIA': 'f',
        'LAIS': 'f',
        'LETICIA': 'f',
        'LIGIA': 'f',
        'MAITHE': 'f',
        'MARIANGELA': 'f',
        'MARINEIA': 'f',
        'MONICA': 'f',
        'NAIADY': 'f',
        'NATHALIA': 'f',
        'NATHALLI': 'f',
        'STHEFANIE': 'f',
        'TAIZA': 'f',
        'TAMILES': 'f',
        'TAIS': 'f',
        'TASSIANY': 'f',
        'TATIANY': 'f',
        'THASSIA': 'f',
        'VERONICA': 'f',
    }

# Expressões podem ser adicionadas aqui para ignorar nomes de voluntários
# Ignorar implica em não fazer análise de gênero, recorrência, retenção e evasão
nomesExcluidos = [re.compile(expressao, re.I) for expressao in [
    r'confirmou',
]]

# A ordem em que os cursos aparecem no dicionário é importante, visto que a busca respeita essa ordem
# Exemplo: "educação física" deve aparecer antes de "física"
cursos = dict((curso, re.compile(expressao, re.I)) for curso, expressao in {
    'Engenharia elétrica': r'el[eé]trica',
    'Psicologia': r'psico',
    'Comunicação social: jornalismo': r'jornal',
    'Medicina': r'medicina|fmrp',
    'Mestrando': r'mestrado|mestrando',
    'Ciência da computação': r'ci[êe]ncias?\s+da\s+computa[cç][aã]o|bcc',
    'Engenharia mecânica': r'mec[aâ]nica',
    'Engenharia de produção': r'produ[cç][aã]o',
    'Engenharia civil': r'civil',
    'Economia Empresarial e Controladoria': r'ecec',
    'Não universitário': r't[eé]cnic[oa]|n[aã]o\s+cursante|completo|etec|trabalho|profissional|convidad[ao]|extern[ao]|palestra|volunt[aá]ri[ao]|nenhum|socorrista|cursinho|vestibula|nutricionista|enfermeira|formad[oa]|consultora|decoradora|estudante|fiscal|terapeuta|banc[aá]ria|psic[oó]log[ao]|assessora|empres[áa]ri[ao]|noite|professor|desempregad[ao]|mãe|graduad[ao]',
    'Meteorologia': r'meteoro',
    'Educação física': r'(educa[çc][ãa]o|ed\.?)\s+f[íi]sica',
    'Física': r'f[ií]sica',
    'Doutorando': r'doutorado',
    'Ciências biológicas': r'biologia|biol[oó]gicas|^bio',
    'Química': r'qu[íi]mica',
    'Administração': r'adm',
    'Música': r'^m[úu]sica',
    'Matemática aplicada a negócios': r'^man|neg[óo]cio',
    'Engenharia química': r'engenharia\s+qu[íi]mica|eng\s+qu[íi]mica',
    'Fisioterapia': r'fisio',
    'Ciências contábeis': r'cont',
    'Economia': r'econo',
    'Pedagogia': r'^pedago',
    'Biblioteconomia e Ciência da Informação': r'^BCI',
    'Universitário: curso não informado': r'^unaerp|cultura|ufpa|ffclrp|^unesp|^integral\s+manh[aã]|^fea',
    'Pós graduando': r'p[óo]s\s+gradua[çc][ãa]o',
    'Agronomia': r'agro',
    'Análise e desenvolvimento de sistemas': r'an[áa]lise',
    'Arquitetura': r'arq',
    'Artes visuais': r'artes',
    'Biotecnologia': r'^biotecnologia',
    'Ciências biomédicas': r'ci[eê]ncias\s+biom[eé]dicas',
    'Comunicação social: radialismo': r'rtv|radialismo|r[aá]dio\s+e\s+tv',
    'Dança, grafiti e teatro': r'teatro',
    'Design': r'design',
    'Direito': r'^direito',
    'Ecologia': r'^ecologia',
    'Enfermagem': r'enfermagem|eerp',
    'Engenharia ambiental': r'amb',
    'Engenharia de biossistemas': r'biossistemas',
    'Engenharia da computação': r'engenharia\s+d[ae]\s+computa[cç][aã]o',
    'Engenharia florestal': r'florestal',
    'Farmácia': r'^farm[áa]cia|fcfrp',
    'Filosofia': r'^filo',
    'Fonoaudiologia': r'^fono',
    'Genética': r'gen[ée]tica',
    'Informática biomédica': r'inform[áa]tica\s+biom[eé]dica|^ibm',
    'Letras': r'^letras',
    'Marketing': r'marketing|mkt',
    'Nutrição e metabolismo': r'nutri[çc][ãa]o',
    'Medicina veterinária': r'veterin[áa]ria',
    'Teologia': r'^teologia',
    'Terapia ocupacional': r'ocupacional|t.o',
    'Odontologia': r'^odonto|forp',
    'Publicidade e propaganda': r'publicidade|pp',
    'Recursos humanos': r'recursos\s+humanos|rh',
    'Relações públicas': r'rela[cç][oõ]es\s+p[uú]blicas|rp',
    'Serviço social': r'social',
    'Sistemas de informação': r'sistemas|^b?si$',
}.items())
listaCursos = [curso for curso, _ in cursos.items()]

loteGeneros = {}
dfs = {}
desc = pd.DataFrame()
pessoas = pd.DataFrame(columns = [Colunas.ID.Descrição, Colunas.Nome.Descrição, Colunas.RG.Descrição, Colunas.CPF.Descrição, Colunas.Quantidade.Descrição, Colunas.Retentora.Descrição, Colunas.Evasora.Descrição, Colunas.Curso.Descrição, Colunas.Gênero.Descrição])
lastID = 0

def createDir(path):
    '''Cria o diretório do caminho indicado, caso não exista.'''
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        if debug: print('Criando diretório ' + directory)
        try:
            os.makedirs(directory, exist_ok = True)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def output(dataframe, path):
    '''Salva o dataframe como csv no caminho indicado.'''
    filename = "../Saida/" + path + "/output.csv"
    createDir(filename)
    if debug: print('Escrevendo ' + path + '/output.csv')
    dataframe.to_csv(filename, index = False, float_format = '%.f')

def incluiAcao(path):
    '''
    A partir de um arquivo csv, faz a inclusão da ação.

    Nenhuma análise é feita e nenhum arquivo é gerado.

    São preenchidas as variáveis globais `dfs`, `pessoas` e `loteGeneros`.
    '''
    ids = []
    if debug: print('Lendo ' + path + ".csv")
    df = pd.read_csv("../Dados/" + path + ".csv", true_values = ['Sim'], false_values = ['Não'])

    if debug: print('Removendo colunas desnecessárias')
    df = df.loc[:, df.columns.str.contains('|'.join([coluna.Expressão for coluna in Colunas if coluna.Expressão]), case = False)]

    # Renomeia colunas para que fiquem de forma homogêna, seguindo a propriedade `Descrição`
    if debug: print('Renomeando colunas')
    def columnIterator():
        '''Retorna apenas as colunas que correspondem a alguma das expressões em `Colunas`.'''
        for coluna in Colunas:
            if coluna.Expressão:
                for col in df.columns:
                    if re.search(coluna.Expressão, col, re.I):
                        yield (col, coluna.Descrição)
                        break

    df.rename(columns = dict(columnIterator()), inplace = True)

    if debug: print('Limpando valores')
    df.replace(r'\t', ' ', regex = True, inplace = True) # Substitui tabs por espaços
    df.replace(r'\s{2,}', ' ', regex = True, inplace = True) # Remove espaços em sequência
    df.replace(r'^\s+|\s+$', '', regex = True, inplace = True) # Leading and trailing trimming
    df.replace(r'^$', None, regex = True, inplace = True) # Transforma vazio em None

    if debug: print('Removendo linhas totalmente em branco')
    df.dropna(axis = 'index', how = 'all', inplace = True)

    # Após remover linhas e colunas não desejadas, refaz os índices
    if debug: print('Refazendo índices')
    df.reset_index(drop = True, inplace = True)

    if debug: print('')
    for i in df.index:
        temNome = False
        if Colunas.Nome.Descrição in df:
            value = df.at[i, Colunas.Nome.Descrição]
            if pd.isnull(value):
                if debug: print('Sem nome')
            elif any([reg.search(value) for reg in nomesExcluidos]):
                df.at[i, Colunas.Nome.Descrição] = None
                if debug: print('Sem nome')
            else:
                # Remove 'pipes' do nome, pois é utilizado como separador na análise de recorrência
                value = re.sub(r'\|', '', value)
                if value == '':
                    df.at[i, Colunas.Nome.Descrição] = None
                    if debug: print('Sem nome')
                else:
                    temNome = True
                    nome = df.at[i, Colunas.Nome.Descrição] = value
                    if debug: print(value)
        elif debug: print('Sem nome')

        def validaDocumento(coluna):
            '''Efetua a validação de CPF ou RG, recuperando apénas dígitos (números em qualquer posição e `x` ou `X` na última posição).'''
            if coluna in df:
                value = df.at[i, coluna]
                if pd.notnull(value):
                    try:
                        int(value) # Se já é int, então não tem caracteres especiais...
                        if debug: print(coluna + ': ' + str(value))
                    except ValueError:
                        newValue = re.sub(r'[^0-9xX]|[xX].', '', value) # Remove caracteres especiais do documento (deixa apenas números)
                        df.at[i, coluna] = None if newValue == '' else newValue
                        if debug: print(coluna + ': ' + value + ' -> ' + newValue)

        validaDocumento(Colunas.RG.Descrição)
        validaDocumento(Colunas.CPF.Descrição)

        # Análise de recorrência
        def analiseRecorrencia(*args):
            '''Busca recorrência por correspondência nas colunas indicadas.'''
            def analiseCurso():
                '''Atribuição imediata do curso de acordo com as expressões definidas no cabeçalho do arquivo. Joga exception caso não encontre.'''
                nome = df.at[i, Colunas.Curso.Descrição]
                if pd.isnull(nome):
                    if debug: print('Curso não preenchido')
                else:
                    try:
                        curso = next(curso for curso, reg in cursos.items() if reg.search(nome))
                        if debug: print('Curso: ' + nome + ' -> ' + curso)
                        return curso
                    except StopIteration:
                        raise Exception('Curso desconhecido: ' + nome)

            def analiseGenero(ID):
                '''
                Caso o gênero esteja no dicionário local, a atribuição é imediata.

                Caso contrário, nome é adicionado ao lote a ser buscado após o fim das inclusões na função `computaGeneros`.
                '''
                primeiroNome = nome.split()[0]
                nomeSemAcento = unidecode(primeiroNome.upper())
                genero = generos.get(nomeSemAcento)
                if genero:
                    pessoas.loc[pessoas[Colunas.ID.Descrição] == ID, Colunas.Gênero.Descrição] = genero
                    if debug: print(primeiroNome + ' -> ' + genero)
                else:
                    # Adiciona nome no lote a ser buscado em `computaGeneros`
                    if nomeSemAcento in loteGeneros:
                        if not ID in loteGeneros[nomeSemAcento]:
                            loteGeneros[nomeSemAcento].append(ID)
                    else:
                        loteGeneros[nomeSemAcento] = [ID]

            def buscaColuna(coluna):
                '''Busca recorrência por correspondência na coluna indicada.'''
                if coluna in df:
                    key = df.at[i, coluna]
                    if pd.notnull(key):
                        if computaRecorrencia:
                            nonlocal nome
                            upperName = unidecode(nome.upper()) # Similaridade não considera acentos (a = á) e é case insensitive (a = A)

                        def analiseNomes(pessoa):
                            '''Retorna o grau de similaridade (0-1) e o nome referente à melhor correspondência (maior similaridade) encontrada.'''
                            nomes = pessoa.split('|')
                            values = [jw.similarity(unidecode(nome.upper()), upperName) for nome in nomes]
                            index, value = max(enumerate(values), key = itemgetter(1))
                            return value, nomes[index]

                        if coluna == Colunas.Nome.Descrição:
                            similaridades = pessoas[coluna].map(analiseNomes)
                            matches = pd.Series([similaridade[0] for similaridade in similaridades]) > .96 # Grau de similaridade mínimo aceitável: 96%
                        else:
                            matches = pessoas[coluna] == key

                        if matches.sum() > 1:
                            # Se acontecer com nome, talvez seja interessante aumentar o grau de similaridade mínimo aceitável
                            # Se acontecer com documento, provavelmente é bug
                            raise Exception('Mais de um registro de ' + coluna + ' "' + key + '" encontrado')

                        if matches.any():
                            ID = pessoas.loc[matches, Colunas.ID.Descrição].iloc[0]
                            if coluna == Colunas.Nome.Descrição:
                                similaridade = max(similaridades, key = itemgetter(0))
                            elif computaRecorrencia:
                                similaridade = analiseNomes(pessoas.loc[matches, Colunas.Nome.Descrição].iloc[0])
                            else:
                                similaridade = [1]

                            if similaridade[0] < 1:
                                # Caso a mesma pessoa dê entrada com nomes diferentes, todos são salvos
                                pessoas.loc[matches & (pessoas[Colunas.Nome.Descrição] == ''), Colunas.Nome.Descrição] = nome
                                pessoas.loc[matches & (pessoas[Colunas.Nome.Descrição] != ''), Colunas.Nome.Descrição] += '|' + nome

                            # Se coluna diverge dentre registros da mesma pessoa, o primeiro encontrado tem valor e o resto é ignorado

                            # Curso
                            if computaCursos and Colunas.Curso.Descrição in df and pd.isnull(pessoas.loc[matches, Colunas.Curso.Descrição].iloc[0]):
                                curso = analiseCurso()
                                if curso:
                                    pessoas.loc[matches, Colunas.Curso.Descrição] = curso

                            # Gênero
                            if computaGenero and Colunas.Nome.Descrição in df and pd.isnull(pessoas.loc[matches, Colunas.Gênero.Descrição].iloc[0]):
                                analiseGenero(ID)

                            if debug:
                                print('Recorrência encontrada pelo ' + coluna + f' ({key})')
                                if coluna == Colunas.Nome.Descrição:
                                    print(f'Similaridade: {similaridade[0] * 100:.0f}% (' + similaridade[1] + ')')

                                print(f'ID: {ID:.0f}')

                            pessoas.loc[matches, Colunas.Quantidade.Descrição] += 1
                            pessoas.loc[matches, Colunas.Evasora.Descrição] = path
                            return ID

            for arg in args:
                if arg is not None:
                    ID = buscaColuna(arg)
                    if ID: return ID

            global lastID
            lastID += 1
            if debug: print(f'Recorrência não encontrada. ID atribuído: {lastID:.0f}')
            pessoas.loc[pessoas.shape[0]] = {
                Colunas.ID.Descrição: lastID,
                Colunas.RG.Descrição: df.at[i, Colunas.RG.Descrição] if Colunas.RG.Descrição in df else None,
                Colunas.CPF.Descrição: df.at[i, Colunas.CPF.Descrição] if Colunas.CPF.Descrição in df else None,
                Colunas.Nome.Descrição: df.at[i, Colunas.Nome.Descrição] if temNome else '',
                Colunas.Quantidade.Descrição: 1,
                Colunas.Retentora.Descrição: path,
                Colunas.Evasora.Descrição: path,
                Colunas.Curso.Descrição: analiseCurso() if computaCursos and Colunas.Curso.Descrição in df and pd.notnull(df.at[i, Colunas.Curso.Descrição]) else None,
                Colunas.Gênero.Descrição: None,
            }
            if computaGenero and Colunas.Nome.Descrição in df and pd.notnull(df.at[i, Colunas.Nome.Descrição]):
                analiseGenero(lastID)

            return lastID

        ID = analiseRecorrencia(Colunas.RG.Descrição, Colunas.CPF.Descrição, Colunas.Nome.Descrição if temNome and computaRecorrencia else None)
        df.at[i, Colunas.ID.Descrição] = ID
        ids.append(ID)
        df.at[i, Colunas.Curso.Descrição] = None
        if debug: print('')

    if Colunas.Curso.Descrição in df:
        df[Colunas.Curso.Descrição] = df[Colunas.Curso.Descrição].apply(lambda value: str(value) if pd.notnull(value) else None)

    dfs[path] = df
    if debug: print('')
    return ids

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def computaGeneros():
    '''
    Busca os gêneros de lotes de 10 nomes a partir da variável `loteGeneros` para o dicionário de gêneros e atribuindo o gênero a cada pessoa.

    Busca acontece primeiro para os nomes mais recorrentes. Caso uma pessoa com múltiplos nomes tenha o gênero encontrado, os outros nomes serão ignorados.

    Caso o nome seja ignorado por todas as pessoas referentes, ele deixa de ser buscado.
    '''
    def iteradorLote():
        for nome, IDs in loteGeneros.items():
            matches = pessoas[Colunas.ID.Descrição].isin(IDs)
            if any(pd.isnull(pessoas.loc[matches, Colunas.Gênero.Descrição])):
                yield nome, matches, IDs.__len__()

    for lote in chunks(sorted(iteradorLote(), key = lambda item: item[2], reverse = True), 10):
        retorno = genderize.get([nome for nome, _, _ in lote])
        #country_id = 'br', language_id = 'pt'
        for genero, (nome, matches, _) in zip([genero[0] if genero else 'n' for genero in [genero['gender'] for genero in retorno]], lote):
            pessoas.loc[matches, Colunas.Gênero.Descrição] = generos[nome] = genero
            if debug: print(nome + ' -> ' + genero)

    arquivo = "../Saida/generos.json"
    createDir(arquivo)
    with open(arquivo, 'w') as outfile:
        json.dump(generos, outfile, sort_keys = True, indent = 2)

    if debug:
        print('Salvando Saida/generos.json')
        print('')

def analisaAcao(path, ids):
    '''
    Faz a análise de gênero e cursos da açao a partir do `DataFrame` referente ao caminho indicado em `dfs`.

    Opcionalmente, indica-se `ids` para restringir a análise apenas a certas pessoas (útil para entrada incremental).
    '''
    if debug: print('Analisando ' + path)
    df = dfs[path]
    algum = not incremental or not os.path.exists("../Saida/" + path + "/output.csv")
    if computaCursos or computaGenero:
        for i in df.index:
            pessoa = pessoas[pessoas[Colunas.ID.Descrição] == df.at[i, Colunas.ID.Descrição]]
            ID = pessoa[Colunas.ID.Descrição].item()
            if ids and not ID in ids: continue
            if debug: print(ID)

            # Curso
            if computaCursos and (not Colunas.Curso.Descrição in df or pd.isnull(df.at[i, Colunas.Curso.Descrição])):
               value = pessoa[Colunas.Curso.Descrição].item()
               if pd.isnull(value):
                   if debug: print('Curso não preenchido')
               else:
                   try:
                       df.at[i, Colunas.Curso.Descrição] = value
                       if debug: print('Curso: ' + value)
                       algum = True
                   except StopIteration:
                       raise Exception('Curso desconhecido: ' + value)

            # Gênero
            if computaGenero and (not Colunas.Gênero.Descrição in df or pd.isnull(df.at[i, Colunas.Gênero.Descrição])):
                value = pessoa[Colunas.Gênero.Descrição].item()
                genero = 'n' if pd.isnull(value) else value
                df.at[i, Colunas.Gênero.Descrição] = genero
                if debug: print('Gênero: ' + genero)
                algum = True

            if debug: print('')

    if algum:
        output(df, path)

def get_last(a):
    '''Retorno o índice da última ocorrência de elemento diferente de `None` na coleção `a`.'''
    return get_n_last(a, 1)

def get_n_last(a, n):
    '''Retorna o índice da enésima última ocorrência de elemento diferente de `None` na coleção `a`.'''
    for i, e in enumerate(reversed(a)):
        if e is not None:
            n = n - 1

        if n == 0:
            return len(a) - i - 1

    return -1

def ranking(nome, colunas, legendas):
    posicaoMaxima = 2
    # 0 = Ano
    # 1 = Mês
    # 2 = Cidade
    g = desc[:-1].groupby(['0', '1', '2']).agg( { coluna: np.sum for coluna in colunas })
    cidades = g.index.get_level_values('2').unique()
    for cidade in cidades:
        if debug: print('Gerando gráfico de ranking por ' + nome + ' para ' + cidade)
        dfAux = pd.DataFrame()
        descCidade = desc[desc['2'] == cidade].reset_index(drop = True)
        r = pd.date_range(pd.to_datetime(calendar.month_abbr[int(descCidade.at[0, '1'])] + '-' + descCidade.at[0, '0']), pd.to_datetime(calendar.month_abbr[int(descCidade.iloc[-1]['1'])] + '-' + descCidade.iloc[-1]['0']), freq = 'MS')
        for date in r:
            filtro = g[(g.index.get_level_values('0') == str(date.year)) & (g.index.get_level_values('1') == f'{date.month:02d}') & (g.index.get_level_values('2') == cidade)]
            dfAux = dfAux.append(filtro.apply(lambda x: x.reset_index(drop = True))
                                       .T
                                       .nlargest(posicaoMaxima, columns = 0)
                                       .rank(method = 'first', ascending = False)
                                       .T
                                       if not filtro.empty else pd.Series(),
            ignore_index = True, sort = False)

        #dfAux = dfAux.apply(lambda x: [min(y, posicaoMaxima + 1) for y in x])
        def iterador():
            for categoria in dfAux.columns:
                yield legendas[colunas.index(categoria)], dfAux[categoria].values

        df = pd.DataFrame({
            legenda: valor for legenda, valor in iterador()
        }, index = r)
        arquivo = "../Saida/" + cidade + "/raking" + nome
        createDir(arquivo)
        notNulls = ~df.isnull().all(axis = 1) # Colunas com pelo menos um valor
        df[notNulls] = df[notNulls].fillna(posicaoMaxima + 1) # Preenche linhas não vazias com valor fixo
        # Não precisa de fill, pois o período termina na última ação da cidade
        #df.fillna(method = 'pad', inplace = True) # Preenche linhas vazias com valor anterior
        # Não precisa de backfill, pois o período começa na primeira ação da cidade
        #df.fillna(method = 'bfill', inplace = True) # Preenche linhas vazias com próximo valor (caso a(s) primeira(s) seja(m) vazia(s))
        # Não precisa limitar a área, pois é garantido que o primeiro e último estejam preenchidos (vide comentários acima)
        df.interpolate(method = 'linear', inplace = True)
        df.to_csv(arquivo + ".csv", index = False, float_format = '%.f')
        ax = df.plot.line(figsize = (32 / 3, 6),
                          legend = False)
        #ax = df.plot.line()
        ax.set_xlabel('Data')
        ax.set_ylabel('Posição')
        posicoes = [str(x) for x in range(1, posicaoMaxima + 1)]
        posicoes.insert(0, '')
        posicoes.append(str(posicaoMaxima + 1) + '+')
        ax.set_yticks(range(0, posicaoMaxima + 2))
        ax.set_yticklabels(posicoes)
        ax.set_ylim(0.5, posicaoMaxima + 1.5)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.title('Ranking por ' + nome + ' em ' + cidade, weight = 'bold')
        #plt.legend()
        plt.savefig(arquivo + ".png")
        plt.close()

def area(nome, colunas, legendas, cores = None):
    # 0 = Ano
    # 1 = Mês
    # 2 = Cidade
    g = desc[:-1].groupby(['0', '1', '2']).agg({ c: np.sum for c in colunas + [Colunas.Válidos.Descrição] })
    cidades = g.index.get_level_values('2').unique()
    for cidade in cidades:
        if debug: print('Gerando gráfico de área de ' + nome + ' para ' + cidade)
        descCidade = desc[desc['2'] == cidade].reset_index(drop = True)
        r = pd.date_range(pd.to_datetime(calendar.month_abbr[int(descCidade.at[0, '1'])] + '-' + descCidade.at[0, '0']), pd.to_datetime(calendar.month_abbr[int(descCidade.iloc[-1]['1'])] + '-' + descCidade.iloc[-1]['0']), freq = 'MS')
        def iterador():
            for coluna in colunas:
                def iterador2():
                    for date in r:
                        filtro = g[(g.index.get_level_values('0') == str(date.year)) & (g.index.get_level_values('1') == f'{date.month:02d}') & (g.index.get_level_values('2') == cidade)]
                        if len(filtro) > 0:
                            validos = filtro[Colunas.Válidos.Descrição][0]
                            if validos > 0:
                                yield 100 * filtro[coluna][0] / validos
                            else:
                                yield None
                        else:
                            yield None

                yield coluna, list(iterador2())

        df = pd.DataFrame({
            legendas[colunas.index(coluna)]: valor for coluna, valor in iterador() if any(valor) > 0
        }, index = r)
        arquivo = "../Saida/" + cidade + "/area" + nome
        createDir(arquivo)
        df.to_csv(arquivo + ".csv", index = False, float_format = '%.f')
        # Não precisa limitar a área, pois é garantido que o primeiro e último estejam preenchidos
        df.interpolate(method = 'linear', inplace = True)
        ax = df.plot.area(figsize = (8, 6),
                          color = [cores[legendas.index(col)] for col in df.columns] if cores else None)
        ax.set_xlabel('Data')
        ax.set_ylabel('Voluntários (%)')
        ax.set_ylim(top = 100)
        plt.title('Área por ' + nome + ' em ' + cidade, weight = 'bold')
        plt.savefig(arquivo + ".png")
        plt.close()

def retencaoEvasao(tipo):
    # O limite superior da data poderia desconsiderar a data da última ação de cada cidade antes de pegar a data da última ação.
    # Isso faria com o que o limite superior pudesse ser ligeiramente menor, possivelmente reduzindo espaço em branco no canto direito do gráfico.
    # Isso vale para retenção. Para evasão, seria limite inferior em vez de superior, primeira ação em vez de última ação e canto esquerdo em vez de direito.
    # Entretanto, não sei fazer isso =)
    r = pd.date_range(pd.to_datetime(calendar.month_abbr[int(desc.at[0, '1'])] + '-' + desc.at[0, '0']), pd.to_datetime(calendar.month_abbr[int(desc.iloc[-2]['1'])] + '-' + desc.iloc[-2]['0']), freq = 'M')
    def iterador():
        # 0 = Ano
        # 1 = Mês
        # 2 = Cidade
        g = desc[:-1].groupby(['0', '1', '2']).agg({ tipo: np.average })
        for cidade in g.index.get_level_values('2').unique():
            def iterador2():
                for date in r:
                    retencoes = g[(g.index.get_level_values('0') == str(date.year)) & (g.index.get_level_values('1') == f'{date.month:02d}') & (g.index.get_level_values('2') == cidade)][tipo]
                    yield retencoes[0] if retencoes.count() > 0 else None

            retencoes = list(iterador2())
            retencoes[get_last(retencoes)] = None
            yield cidade, retencoes

    if debug: print('Gerando gráfico de ' + tipo)
    df = pd.DataFrame({
        cidade: retencoes for cidade, retencoes in iterador()
    }, index = r)
    arquivo = "../Saida/Descritivo/" + tipo
    createDir(arquivo)
    df.to_csv(arquivo + ".csv", index = False, float_format = '%.f')
    df.interpolate(method = 'linear', limit_area = 'inside', inplace = True) # Interpola pontos, permitindo que não exista "gaps" entre pontos quando não há ação no mês para aquela cidade
    ax = df.plot.line()
    ax.set_ylim(bottom = 0) # Gráfico começa do zero (%), pois não existe porcentagem negativa (limite padrão admite pequena parte negativa)
    if max(np.max(df)) > 50: # Caso o maior índice seja maior que 50%
        ax.set_ylim(top = 100) # Então o limite superior do gráfico é 100 (%), pois não existe porcentagem acima disso e fica mais claro que se trata de porcentagem (em vez de um valor menor)

    ax.set_xlabel('Data')
    ax.set_ylabel('Voluntários (%)')
    plt.grid(True)
    plt.title(tipo + ' por ação', weight = 'bold')
    plt.savefig(arquivo + ".png")
    plt.close()

def setores(agrupador, nome, colunas, legenda, cores = None):
    def preparaGrafico():
        for grupo in agrupador():
            count = grupo.__len__()
            matches = desc[:-1].groupby([str(x) for x in grupo]).agg({ c: np.sum for c in colunas + [Colunas.Válidos.Descrição] }) if count > 0 else desc
            yield count, matches

    for count, matches in preparaGrafico():
        for i in matches.index:
            df = pd.DataFrame({
                'Legenda': legenda,
                nome: [matches.at[i, col] for col in colunas]
            }, index = legenda).sort_values([nome, 'Legenda'], ascending = False)
            if df[nome].iloc[0] != 0:
                acao = '/'.join([i] if count == 1 else i) if count > 0 else matches.at[i, Colunas.Ação.Descrição]
                if acao == 'Total':
                    acao = 'Descritivo'

                if debug: print('Gerando gráfico de setores de ' + nome + ' para ' + acao)
                df = df[df[nome] != 0]
                muitos = df.shape[0] > 6 # Se tiver mais de 6 valores, então utiliza-se apenas 5, sendo o sexto o "outros", cujo valor é a soma dos restantes
                if (muitos):
                    outros = pd.DataFrame({
                        'Legenda': ['Outros'],
                        nome: [df[nome][5:].sum()]
                    }, index = ['Outros'])
                    df = pd.concat([df[:5], outros])

                diff = matches.at[i, Colunas.Válidos.Descrição] - df[nome].sum()
                if diff < 0: raise Exception('Quantidade de voluntários válidos é menor do que somatória de ' + legenda + ' ' + nome) # Se acontecer, é bug
                if diff > 0:
                    semresposta = pd.DataFrame({
                        'Legenda': ['Sem resposta'],
                        nome: [diff]
                    }, index = ['Sem resposta'])
                    df = pd.concat([df, semresposta])

                arquivo = "../Saida/" + acao + "/" + nome
                createDir(arquivo)

                df.to_csv(arquivo + ".csv", index = False, float_format = '%.f')
                wedges, texts, autotexts = plt.pie(df[nome],
                                                   colors = [cores[legenda.index(df.at[i, 'Legenda'])] for i in df.index] if cores else None,
                                                   autopct = lambda p : '{:.0f}%'.format(p),
                                                   labels = df['Legenda'] if not muitos else None)
                for wedge in wedges:
                    wedge.set_edgecolor('white')

                for text in autotexts: # Deixa os textos "invisíveis" (mesma cor do fundo)
                    text.set_color('white')

                if muitos:
                    plt.legend(df['Legenda'],
                               loc = 'lower left',
                               bbox_to_anchor = (-0.35, -0.1),
                               handlelength = 1,
                               labelspacing = 0.1)

                plt.title(nome, weight = 'bold')
                plt.savefig(arquivo + ".png")
                plt.close()

def descritivo():
    if debug: print('Calculando dados descritivos')
    i = 0
    for (path, df) in dfs.items():
        desc.at[i, Colunas.Ação.Descrição] = path
        desc.at[i, Colunas.Válidos.Descrição] = df.shape[0]
        if computaGenero:
            desc.at[i, 'Qtde. homens'] = df[df[Colunas.Gênero.Descrição] == 'm'].shape[0] if Colunas.Gênero.Descrição in df else 0
            desc.at[i, 'Qtde. mulheres'] = df[df[Colunas.Gênero.Descrição] == 'f'].shape[0] if Colunas.Gênero.Descrição in df else 0
            desc.at[i, 'Qtde. não informado'] = df[df[Colunas.Gênero.Descrição] == 'n'].shape[0] if Colunas.Gênero.Descrição in df else 0

        if computaCursos:
            for curso in listaCursos:
                desc.at[i, 'Qtde. ' + curso] = df[df[Colunas.Curso.Descrição] == curso].shape[0] if Colunas.Curso.Descrição in df else 0

        if computaRecorrencia:
            pessoasNovas = pessoas[pessoas[Colunas.Retentora.Descrição] == path].shape[0]
            pessoasVelhas = pessoas[pessoas[Colunas.Retentora.Descrição] != path].shape[0]
            desc.at[i, Colunas.Retenção.Descrição] = pessoas[(pessoas[Colunas.Retentora.Descrição] == path) & (pessoas[Colunas.Quantidade.Descrição] > 1)].shape[0] / pessoasNovas * 100 if pessoasNovas > 0 else 0
            desc.at[i, Colunas.Evasão.Descrição] = pessoas[(pessoas[Colunas.Evasora.Descrição] == path) & (pessoas[Colunas.Quantidade.Descrição] > 1)].shape[0] / pessoasVelhas * 100 if pessoasVelhas > 0 else 0

        i += 1

    if computaRecorrencia:
        mediaRetencao = desc.loc[:, Colunas.Retenção.Descrição].mean()
        mediaEvasao = desc.loc[:, Colunas.Evasão.Descrição].mean()

    desc.loc[i, :] = desc.sum(axis = 0)
    if computaRecorrencia:
        desc.at[i, Colunas.Retenção.Descrição] = mediaRetencao
        desc.at[i, Colunas.Evasão.Descrição] = mediaEvasao

    desc.at[i, Colunas.Ação.Descrição] = 'Total'
    output(desc, 'Descritivo')

def geraGraficos():
    g = desc.at[0, Colunas.Ação.Descrição].split('/').__len__() - 1
    r = range(0, g)
    def agrupador():
        '''
        Retorna toda a análise combinatória:
        - Nenhum agrupamento (ação por ação);
        - Ano;
        - Mês;
        - Cidade;
        - Ano, Mês;
        - Ano, Cidade;
        - Ano, Mês, Cidade.
        '''
        yield list()
        for i in range(1, g):
            for combination in itertools.combinations(r, i):
                yield list(combination)

    s = desc[Colunas.Ação.Descrição].str.split('/').str
    for i in r:
        desc[str(i)] = s[i]

    if computaGenero:
        setores(agrupador, 'Gênero', ['Qtde. homens', 'Qtde. mulheres', 'Qtde. não informado'], ['Homens', 'Mulheres', 'Não informado'], ['#1f77b4', '#d62728', '#32cd32'])
        area('Gênero', ['Qtde. mulheres', 'Qtde. não informado', 'Qtde. homens'], ['Mulheres', 'Não informado', 'Homens'], ['#d62728', '#32cd32', '#1f77b4'])

    if computaCursos:
        setores(agrupador, 'Cursos', ['Qtde. ' + curso for curso in listaCursos], listaCursos)
        ranking('Cursos', ['Qtde. ' + curso for curso in listaCursos], listaCursos)

    if computaRecorrencia:
        retencaoEvasao(Colunas.Retenção.Descrição)
        retencaoEvasao(Colunas.Evasão.Descrição)

def recorrencia():
    if debug: print('Fazendo análise de recorrência')
    pessoas.sort_values(Colunas.Quantidade.Descrição, ascending = False, inplace = True)
    output(pessoas, 'Recorrencia')

def continuidade():
    '''Faz a leitura diretamente da saída para evitar reprocessamento'''
    for file in Path("../Saida").glob('**/output.csv'):
        path = str(file)
        if debug: print('Carregando ' + '/'.join(path.split('\\')[2:]))
        try:
            df = pd.read_csv(file, true_values = ['Sim'], false_values = ['Não'])
        except Exception as ex:
            if debug:
                print('Ocorreu um erro ao ler o arquivo ' + path)
                print(ex)

            continue

        path = '/'.join(path.split('\\')[2:-1])
        if 'Descritivo' in path:
            global desc
            desc = df
        elif 'Recorrencia' in path:
            global pessoas
            pessoas = df
            global lastID
            lastID = pessoas[Colunas.ID.Descrição].max()
        else:
            dfs[path] = df

if debug: print('Working directory: ' + os.getcwd())
caminhoInicial = "../Dados/"
ids = []
acoes = [str(file)[caminhoInicial.__len__():-4].replace('\\', '/') for file in Path(caminhoInicial).glob('**/*.csv')]
# Estrutura se dá por:
# Ano
# ╚Mês
#  ╚Cidade
#   ╚Ação
if entrada:
    if incremental:
        continuidade()

    for acao in acoes:
        if not incremental or acao not in dfs:
            ids.extend(incluiAcao(acao))

    if computaGenero:
        computaGeneros()

    for path in dfs:
        analisaAcao(path, ids)
else:
    continuidade()

if ids:
    descritivo()
    recorrencia()

if graficos:
    geraGraficos()
