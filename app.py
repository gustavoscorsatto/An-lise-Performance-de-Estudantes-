import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import joblib 
import os 

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Performance dos Estudantes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARREGAMENTO DO MODELO E COLUNAS ---
@st.cache_resource
def carregar_modelo_e_colunas():
    if os.path.exists('modelo_random_forest.pkl') and os.path.exists('colunas_modelo.pkl'):
        modelo = joblib.load('modelo_random_forest.pkl')
        colunas = joblib.load('colunas_modelo.pkl')
        return modelo, colunas
    return None, None

modelo, colunas_modelo = carregar_modelo_e_colunas()



# --- CARREGAMENTO DOS DADOS ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv('student_data_tratados.csv')
    return df

df = carregar_dados()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("Painel de Controle")
st.sidebar.markdown("Crie filtros personalizados para explorar os dados.")

# --- FILTRO DINÂMICO ---
st.sidebar.header("Filtro Dinâmico")
colunas_uteis = [col for col in df.columns if col not in ['Nota_1T', 'Nota_2T', 'Nota_3T_Final', 'media_final']]
colunas_para_filtrar = st.sidebar.multiselect(
    "Selecione as colunas para filtrar:",
    options=colunas_uteis,
    default=['Idade', 'Sexo', 'Internet']
)

filtros_aplicados = {}

for coluna in colunas_para_filtrar:
    st.sidebar.markdown(f"**Filtro para `{coluna}`**")
    if pd.api.types.is_numeric_dtype(df[coluna]):
        min_val, max_val = int(df[coluna].min()), int(df[coluna].max())
        valor_selecionado = st.sidebar.slider(f"Intervalo para {coluna}", min_val, max_val, (min_val, max_val))
        filtros_aplicados[coluna] = valor_selecionado
    else:
        opcoes = df[coluna].unique()
        valores_selecionados = st.sidebar.multiselect(f"Valores para {coluna}", opcoes, default=opcoes)
        filtros_aplicados[coluna] = valores_selecionados
st.sidebar.markdown("---")

# --- APLICANDO OS FILTROS ---
df_filtrado = df.copy()
for coluna, valor in filtros_aplicados.items():
    if pd.api.types.is_numeric_dtype(df_filtrado[coluna]):
        df_filtrado = df_filtrado[df_filtrado[coluna].between(valor[0], valor[1])]
    else:
        df_filtrado = df_filtrado[df_filtrado[coluna].isin(valor)]


# --- LAYOUT PRINCIPAL ---
st.title("🎓 Análise de Performance dos Estudantes")
st.markdown("Bem-vindo ao painel de análise de desempenho acadêmico.")

# --- KPIs ---
st.divider()
col1, col2 = st.columns(2)
total_alunos = df_filtrado.shape[0]
media_geral = df_filtrado['media_final'].mean() if total_alunos > 0 else 0.0
col1.metric("Total de Alunos (Filtrado)", value=total_alunos)
col2.metric("Média Geral da Nota (Filtrada)", value=f"{media_geral:.2f}")
st.divider()

# --- ABAS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Análise Exploratória",
    "🤖 Previsão de Desempenho",
    "👥 Equipe de Desenvolvedores"
])

# Análise Exploratória dos Dados 
with tab1:
    st.header("Análise Exploratória dos Dados")
    if total_alunos == 0:
        st.warning("Nenhum dado corresponde aos filtros selecionados. Por favor, ajuste os filtros na barra lateral.")
    else:
        st.markdown("Explore a relação entre diferentes variáveis e o desempenho dos alunos.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribuição das Notas Médias Finais")
            fig_hist_notas = px.histogram(df_filtrado, x='media_final', nbins=20, title="Distribuição da Nota Média Final", color_discrete_sequence=['#0083B8'])
            st.plotly_chart(fig_hist_notas, use_container_width=True)

        with col2:
            st.subheader("Notas por Sexo")
            fig_box_sexo = px.box(df_filtrado, x='Sexo', y='media_final', color='Sexo', title="Nota Média por Sexo", labels={'Sexo': 'Sexo', 'media_final': 'Nota Média Final'})
            st.plotly_chart(fig_box_sexo, use_container_width=True)

        st.divider()

        st.subheader("Correlação entre Variáveis e a Nota Média")
        opcoes_scatter = ['Tempo_Estudo', 'Falhas_Anteriores', 'Faltas', 'Idade', 'Educacao_Mae', 'Educacao_Pai', 'Tempo_Livre', 'Saida_Amigos']
        var_x = st.selectbox('Selecione uma variável para o eixo X:', options=opcoes_scatter)
        var_y = 'media_final'

        fig_scatter = px.scatter(df_filtrado, x=var_x, y=var_y,
                                 title=f'Relação entre {var_x.replace("_", " ")} e Nota Média Final',
                                 labels={var_x: var_x.replace("_", " "), var_y: 'Nota Média Final'},
                                 color='Idade',
                                 color_continuous_scale=px.colors.sequential.Viridis,
                                 trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)


# --- ABA DE PREVISÃO ATUALIZADA ---
with tab2:
    st.header("Previsão de Desempenho do Aluno")

    classificador_rf = RandomForestClassifier()
    classificador_rf.fit(X_treino, y_treino)

    joblib.dump(classificador_rf, 'modelo_random_forest.pkl')
    joblib.dump(atrib_pre_Padronizacao.columns.tolist(), 'colunas_modelo.pkl')
    
    if modelo is None or colunas_modelo is None:
        st.error("Arquivos do modelo (`modelo_random_forest.pkl` e `colunas_modelo.pkl`) não encontrados. Por favor, execute o script de treinamento primeiro.")
    else:
        st.markdown("Insira os dados de um aluno para prever sua faixa de desempenho.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Características Gerais")
            input_features = {}
            input_features['Idade'] = st.slider('Idade', 15, 22, 17)
            input_features['Sexo'] = st.selectbox('Sexo', options=df['Sexo'].unique())
            input_features['Endereco'] = st.selectbox('Endereço (U: Urbano, R: Rural)', options=df['Endereco'].unique())
            input_features['Tempo_Estudo'] = st.select_slider('Tempo de Estudo Semanal', options=[1, 2, 3, 4], format_func=lambda x: f'{x} ({ {1: "< 2h", 2: "2-5h", 3: "5-10h", 4: "> 10h"}[x] })')
            input_features['Falhas_Anteriores'] = st.number_input('Número de Falhas Anteriores', min_value=0, max_value=5, step=1)
            input_features['Faltas'] = st.number_input('Número de Faltas', min_value=0, max_value=93, step=1)
            input_features['Internet'] = st.selectbox('Possui acesso à Internet?', options=df['Internet'].unique())


        with col2:
            st.subheader("Ambiente Familiar")
            input_features['Educacao_Mae'] = st.selectbox('Nível de Educação da Mãe', options=sorted(df['Educacao_Mae'].unique()))
            input_features['Educacao_Pai'] = st.selectbox('Nível de Educação do Pai', options=sorted(df['Educacao_Pai'].unique()))
            input_features['Profissao_Mae'] = st.selectbox('Profissão da Mãe', options=df['Profissao_Mae'].unique())
            input_features['Profissao_Pai'] = st.selectbox('Profissão do Pai', options=df['Profissao_Pai'].unique())
            input_features['Tamanho_Familia'] = st.selectbox('Tamanho da Família (GT3: Maior que 3, LE3: Menor ou igual a 3)', options=df['Tamanho_Familia'].unique())
            input_features['Status_Pais'] = st.selectbox('Status dos Pais (T: Juntos, A: Separados)', options=df['Status_Pais'].unique())
            
        if st.button('Prever Faixa de Desempenho', type="primary"):
            df_input = pd.DataFrame([input_features])
            
            df_input_encoded = pd.get_dummies(pd.concat([df, df_input], ignore_index=True)).tail(1)

            df_input_aligned = df_input_encoded.reindex(columns=colunas_modelo, fill_value=0)
            
            previsao_array = modelo.predict(df_input_aligned) 
            
            faixa_prevista = previsao_array[0] 
            
            st.success(f"A faixa de desempenho prevista para este aluno é: **{faixa_prevista}**")

            if faixa_prevista == 'Pessima' or faixa_prevista == 'Ruim':
                st.error("Atenção: Risco de desempenho insuficiente. Apoio pedagógico pode ser recomendado.")
            elif faixa_prevista == 'Excelente':
                st.balloons()
                st.info("Parabéns! Desempenho excelente previsto.")


with tab3:
    st.header("Equipe de Desenvolvedores")
    st.divider()
    st.markdown("""
    - **Nome:** Beatriz Yanagihara -**Área:** Marketing
    - **Nome:** Gustavo Scorsatto - **Área:** Projetos
    - **Nome:** Lucas Saad - **Área:** Training
    - **Nome:** Thomas Araújo - **Área:** RH
    - **Nome:** Tito Amado - Área:** RH
    """)











