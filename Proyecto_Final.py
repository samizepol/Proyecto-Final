import pandas as pd
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
import os
from dotenv import load_dotenv
load_dotenv('APIS_KEY.env')

warnings.filterwarnings('ignore')

class AnalizarTecnologias:
    def __init__(self):
        self.stackoverflow_base_url = "https://api.stackexchange.com/2.3"
        self.github_search_url = "https://api.github.com/search/repositories"
        
        # Mapeo extendido de tecnologías
        self.technology_mapping = {
            'javascript': 'JavaScript', 'js': 'JavaScript',
            'python': 'Python', 'py': 'Python',
            'java': 'Java',
            'c++': 'C++', 'cpp': 'C++',
            'c#': 'C#', 'csharp': 'C#',
            'typescript': 'TypeScript', 'ts': 'TypeScript',
            'go': 'Go', 'golang': 'Go',
            'rust': 'Rust',
            'php': 'PHP',
            'ruby': 'Ruby',
            'swift': 'Swift',
            'kotlin': 'Kotlin',
            'react': 'React', 'reactjs': 'React',
            'vue': 'Vue.js', 'vuejs': 'Vue.js',
            'angular': 'Angular',
            'node': 'Node.js', 'nodejs': 'Node.js',
            'django': 'Django',
            'flask': 'Flask',
            'spring': 'Spring',
            'laravel': 'Laravel',
            'tensorflow': 'TensorFlow',
            'pytorch': 'PyTorch',
            'docker': 'Docker',
            'kubernetes': 'Kubernetes'
        }
        
        # rastrea tecnologías históricamente
        self.target_technologies = [
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'TypeScript', 'Rust'
        ]

    def get_historical_stackoverflow_data(self, years: int = 3) -> pd.DataFrame:
        """
        Obtiene datos históricos reales de Stack Overflow por trimestres,
        contando todas las preguntas por tecnología en cada periodo.
        """
        print(f"Obteniendo datos históricos de Stack Overflow para {years} años...")

        all_data = []
        current_date = datetime.now()

        stack_token = os.getenv("STACK_TOKEN")
        if not stack_token:
            print("Advertencia: STACK_KEY no encontrada. Se usará modo sin autenticación.")


        for year_back in range(years):
            for quarter in range(4):
                quarter_start = datetime(current_date.year - year_back, quarter * 3 + 1, 1)

                # Fin del trimestre
                if quarter == 3:
                    quarter_end = datetime(current_date.year - year_back + 1, 1, 1) - timedelta(days=1)
                else:
                    quarter_end = datetime(current_date.year - year_back, (quarter + 1) * 3 + 1, 1) - timedelta(days=1)

                # Salta periodos futuros
                if quarter_start > current_date:
                    continue
                if quarter_end > current_date:
                    quarter_end = current_date

                for tech in self.target_technologies:
                    try:
                        print(f"- {tech} ({quarter_start.year}-Q{quarter+1}) ...", end=' ', flush=True)
                        total = 0
                        page = 1

                        while True:
                            params = {
                                'page': page,
                                'pagesize': 100,
                                'fromdate': int(quarter_start.timestamp()),
                                'todate': int(quarter_end.timestamp()),
                                'order': 'desc',
                                'sort': 'creation',
                                'tagged': tech.lower(),
                                'site': 'stackoverflow',
                            }

                            if stack_token:
                                params['key'] = stack_token

                            response = requests.get(
                                "https://api.stackexchange.com/2.3/questions",
                                params=params,
                                timeout=20
                            )

                            # Evita parsear JSON vacío
                            if not response.text.strip():
                                print(f"Respuesta vacía (posible límite alcanzado)")
                                break

                            data = response.json()

                            items = data.get('items', [])
                            total += len(items)

                            # Control de fin de páginas
                            if not data.get('has_more', False):
                                break
                            page += 1
                            time.sleep(1.5)  # Previene rate-limit

                        print(f"total: {total}")

                        all_data.append({
                            'technology': tech,
                            'count': total,
                            'date': quarter_start,
                            'quarter': f"{quarter_start.year}-Q{quarter + 1}",
                            'year': quarter_start.year,
                            'quarter_num': quarter + 1,
                            'source': 'Stack Overflow'
                        })

                    except Exception as e:
                        print(f"\nError con {tech} en {quarter_start.date()}: {e}")
                        continue

        df = pd.DataFrame(all_data)
        print(f"\n Datos históricos obtenidos ({len(df)} registros totales).")
        return df

        
    def get_historical_github_data(self, years: int = 3) -> pd.DataFrame:
        """
        Obtiene datos históricos de GitHub por años.
        """
        
        all_data = []
        current_year = datetime.now().year

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN no encontrado en variables de entorno")

        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {github_token}'  
        }

        for year in range(current_year - years + 1, current_year + 1):
            print(f"Procesando año {year}...")

            for tech in self.target_technologies:
                try:
                    query = f"language:{tech.lower()} created:{year}-01-01..{year}-12-31"
                    params = {'q': query, 'per_page': 1}

                    response = safe_get(self.github_search_url, params=params, headers=headers)
                    data = response.json()

                    total = data.get('total_count', 0)
                    if total == 0:
                        print(f"Advertencia: sin datos válidos para {tech} en {year}")
                        continue

                    all_data.append({
                        'technology': tech,
                        'count': total,
                        'date': datetime(year, 1, 1),
                        'quarter': f"{year}-Annual",
                        'year': year,
                        'quarter_num': 1,
                        'source': 'GitHub'
                    })

                    time.sleep(1.2)

                except Exception as e:
                    print(f"Error obteniendo datos GitHub para {tech} en {year}: {e}")
                    continue

        return pd.DataFrame(all_data)
    
    def create_sample_historical_data(self, years: int = 5) -> pd.DataFrame:
        """
        Crea datos de muestra realistas para análisis histórico
        """
        print(f"Generando datos de muestra para {years} años...")
        
        all_data = []
        end_date = datetime.now()
        start_date = datetime(end_date.year - years, 1, 1)
        
        # Tendencias históricas realistas basadas en datos del mercado
        trends_config = {
            'Python': {'base': 1000, 'growth': 1.15, 'volatility': 0.1},
            'JavaScript': {'base': 1200, 'growth': 1.08, 'volatility': 0.08},
            'Java': {'base': 800, 'growth': 0.98, 'volatility': 0.05},
            'C++': {'base': 600, 'growth': 1.02, 'volatility': 0.06},
            'C#': {'base': 700, 'growth': 1.05, 'volatility': 0.07},
            'TypeScript': {'base': 300, 'growth': 1.25, 'volatility': 0.12},
            'Go': {'base': 200, 'growth': 1.18, 'volatility': 0.15},
            'Rust': {'base': 100, 'growth': 1.30, 'volatility': 0.20},
            'PHP': {'base': 500, 'growth': 0.95, 'volatility': 0.04},
            'Ruby': {'base': 400, 'growth': 0.92, 'volatility': 0.05},
            'Swift': {'base': 350, 'growth': 1.10, 'volatility': 0.09},
            'Kotlin': {'base': 250, 'growth': 1.20, 'volatility': 0.11}
        }
        
        current_date = start_date
        quarter = 1
        year = start_date.year
        
        while current_date <= end_date:
            for tech, config in trends_config.items():
                # Calcular trimestre
                quarter_num = (current_date.month - 1) // 3 + 1
                
                # Calcular factor de crecimiento con variación estacional
                quarters_passed = (year - start_date.year) * 4 + quarter_num - 1
                base_growth = config['growth'] ** (quarters_passed / 4)
                
                # Variación estacional (más actividad en Q1 y Q3)
                seasonal_factor = 1 + 0.1 * np.sin(quarters_passed * np.pi / 2)
                
                # Ruido aleatorio
                noise = np.random.normal(0, config['volatility'])
                
                count = int(config['base'] * base_growth * seasonal_factor * (1 + noise))
                
                all_data.append({
                    'technology': tech,
                    'count': max(count, 10),  # Mínimo 10
                    'date': current_date,
                    'quarter': f"{year}-Q{quarter_num}",
                    'year': year,
                    'quarter_num': quarter_num,
                    'source': 'Historical Sample'
                })
            
            # Avanzar al siguiente trimestre
            if quarter_num == 4:
                year += 1
                current_date = datetime(year, 1, 1)
            else:
                current_date = datetime(year, quarter_num * 3 + 1, 1)
        
        return pd.DataFrame(all_data)
    
    def calculate_historical_metrics(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas históricas avanzadas
        """
        if historical_data.empty:
            return pd.DataFrame()
        
        metrics = []
        
        for tech in historical_data['technology'].unique():
            tech_data = historical_data[historical_data['technology'] == tech].sort_values('date')
            
            if len(tech_data) < 4:  # Necesitamos al menos 4 trimestres para análisis significativo
                continue
            
            # Métricas básicas
            initial_count = tech_data['count'].iloc[0]
            final_count = tech_data['count'].iloc[-1]
            total_growth_pct = ((final_count - initial_count) / initial_count) * 100 if initial_count > 0 else 0
            
            # Crecimiento anual compuesto (CAGR)
            years = (tech_data['date'].iloc[-1] - tech_data['date'].iloc[0]).days / 365.25
            cagr = ((final_count / initial_count) ** (1/years) - 1) * 100 if years > 0 and initial_count > 0 else 0
            
            # Tendencia lineal
            x = np.arange(len(tech_data))
            slope, intercept = np.polyfit(x, tech_data['count'], 1)
            r_squared = np.corrcoef(x, tech_data['count'])[0, 1] ** 2
            
            # Volatilidad (desviación estándar de los retornos trimestrales)
            returns = tech_data['count'].pct_change().dropna()
            volatility = returns.std() * 100 if len(returns) > 0 else 0
            
            # Momentum (crecimiento en los últimos 4 trimestres)
            if len(tech_data) >= 5:
                recent_growth = ((tech_data['count'].iloc[-1] - tech_data['count'].iloc[-5]) / 
                               tech_data['count'].iloc[-5]) * 100
            else:
                recent_growth = total_growth_pct
            
            metrics.append({
                'technology': tech,
                'initial_popularity': initial_count,
                'current_popularity': final_count,
                'total_growth_percentage': total_growth_pct,
                'cagr': cagr,
                'trend_slope': slope,
                'r_squared': r_squared,
                'volatility': volatility,
                'recent_growth': recent_growth,
                'data_points': len(tech_data),
                'analysis_period_years': years
            })
        
        return pd.DataFrame(metrics)
    
    def create_growth_comparison_chart(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Crea gráfico de comparación de crecimiento histórico
        """
        fig = go.Figure()
        
        # Barras para CAGR
        fig.add_trace(go.Bar(
            x=metrics_df['technology'],
            y=metrics_df['cagr'],
            name='CAGR (%)',
            marker_color='lightblue',
            text=metrics_df['cagr'].round(2),
            textposition='auto'
        ))
        
        # Línea para crecimiento reciente
        fig.add_trace(go.Scatter(
            x=metrics_df['technology'],
            y=metrics_df['recent_growth'],
            name='Crecimiento Reciente (%)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Comparación de Crecimiento Histórico - CAGR vs Crecimiento Reciente',
            xaxis_title='Tecnología',
            yaxis_title='CAGR (%)',
            yaxis2=dict(
                title='Crecimiento Reciente (%)',
                overlaying='y',
                side='right'
            ),
            xaxis_tickangle=45,
            height=600
        )
        
        return fig
    
    def create_market_share_pie_chart(self, historical_data: pd.DataFrame, year: int = None) -> go.Figure:
        """
        Crea gráfico de pastel mostrando la distribución de mercado por año
        """
        if year is None:
            year = historical_data['year'].max()
        
        # Filtrar datos del año específico
        year_data = historical_data[historical_data['year'] == year]
        
        if year_data.empty:
            print(f"No hay datos para el año {year}")
            return go.Figure()
        
        # Agrupar por tecnología y sumar las menciones
        tech_shares = year_data.groupby('technology')['count'].sum().reset_index()
        tech_shares = tech_shares.sort_values('count', ascending=False)
        
        # Calcular porcentajes
        total = tech_shares['count'].sum()
        tech_shares['percentage'] = (tech_shares['count'] / total) * 100
        
        # Crear gráfico de pastel
        fig = px.pie(
            tech_shares,
            values='count',
            names='technology',
            title=f'Distribución del Mercado de Tecnologías - Año {year}',
            hover_data=['percentage'],
            labels={'count': 'Menciones', 'technology': 'Tecnología'}
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Menciones: %{value}<br>Porcentaje: %{percent}'
        )
        
        fig.update_layout(
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_yearly_ranking_chart(self, historical_data: pd.DataFrame) -> go.Figure:
        """
        Crea un gráfico de ranking anual animado 
        basado en la popularidad total de cada tecnología.
        """
        if historical_data.empty:
            print("No hay datos disponibles para generar el ranking anual.")
            return go.Figure()
        
        # Agrupar datos por año y tecnología
        yearly_ranking = (
            historical_data.groupby(['year', 'technology'])['count']
            .sum()
            .reset_index()
        )

        # Calcular el ranking dentro de cada año
        yearly_ranking['rank'] = yearly_ranking.groupby('year')['count'] \
            .rank(ascending=False, method='first')

        # Crear gráfico de barras animado
        fig = px.bar(
            yearly_ranking.sort_values(['year', 'rank']),
            x='count',
            y='technology',
            color='technology',
            orientation='h',
            animation_frame='year',
            animation_group='technology',
            title='Ranking Anual de Tecnologías (Popularidad Total)',
            labels={'count': 'Popularidad (Menciones)', 'technology': 'Tecnología'},
            range_x=[0, yearly_ranking['count'].max() * 1.2],
            height=650
        )

        fig.update_layout(
            showlegend=False
        )

        return fig

    def create_historical_ranking(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea ranking histórico compuesto
        """
        ranked_df = metrics_df.copy()
        
        # Normalizar métricas para ranking
        metrics_to_rank = ['cagr', 'current_popularity', 'recent_growth', 'r_squared']
        
        for metric in metrics_to_rank:
            ranked_df[f'{metric}_rank'] = ranked_df[metric].rank(ascending=False, method='min')
        
        # Ranking compuesto (ponderado)
        weights = {
            'cagr': 0.35,           # Crecimiento histórico
            'current_popularity': 0.25,  # Popularidad actual
            'recent_growth': 0.25,   # Momentum reciente
            'r_squared': 0.15        # Consistencia de tendencia
        }
        
        ranked_df['composite_rank'] = sum(ranked_df[f'{metric}_rank'] * weight 
                                        for metric, weight in weights.items())
        
        ranked_df = ranked_df.sort_values('composite_rank')
        
        return ranked_df[['technology', 'cagr', 'current_popularity', 'recent_growth', 
                         'r_squared', 'volatility', 'composite_rank']]
    
    def generate_historical_report(self, historical_data: pd.DataFrame, metrics_df: pd.DataFrame):
        """
        Genera reporte histórico completo
        """
        if metrics_df.empty:
            print("No se pudieron calcular métricas suficientes.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_technology_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Datos históricos crudos
            historical_data.to_excel(writer, sheet_name='Datos Históricos', index=False)
            
            # Métricas calculadas
            metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
            
            # Ranking
            ranking_df = self.create_historical_ranking(metrics_df)
            ranking_df.to_excel(writer, sheet_name='Ranking', index=False)
            
            # Resumen ejecutivo
            summary_data = {
                'Métrica': [
                    'Período Analizado',
                    'Tecnologías Analizadas',
                    'Tecnología con Mayor CAGR',
                    'Tecnología Más Popular',
                    'Tecnología Más Volátil',
                    'Tecnología Más Estable',
                    'CAGR Promedio',
                    'Volatilidad Promedio'
                ],
                'Valor': [
                    f"{len(historical_data['year'].unique())} años",
                    len(metrics_df),
                    metrics_df.loc[metrics_df['cagr'].idxmax(), 'technology'],
                    metrics_df.loc[metrics_df['current_popularity'].idxmax(), 'technology'],
                    metrics_df.loc[metrics_df['volatility'].idxmax(), 'technology'],
                    metrics_df.loc[metrics_df['volatility'].idxmin(), 'technology'],
                    f"{metrics_df['cagr'].mean():.2f}%",
                    f"{metrics_df['volatility'].mean():.2f}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen', index=False)
        
        print(f"Reporte histórico guardado como: {filename}")
        return filename
    
    def run_historical_analysis(self, years: int = 5, use_real_data: bool = False):
        """
        Ejecuta análisis histórico completo
        """
        print(f"Iniciando análisis histórico de {years} años...")
        
       # Obtiene los datos
        if use_real_data:
            try:
                stackoverflow_data = self.get_historical_stackoverflow_data(years)
                github_data = self.get_historical_github_data(years)
                historical_data = pd.concat([stackoverflow_data, github_data], ignore_index=True)
            except Exception as e:
                print(f"Error obteniendo datos reales: {e}")
                print("Usando datos de muestra...")
                historical_data = self.create_sample_historical_data(years)
        else:
            historical_data = self.create_sample_historical_data(years)
        
        if historical_data.empty:
            print("No se pudieron obtener datos para el análisis")
            return None
        
        # Calcular métricas
        metrics_df = self.calculate_historical_metrics(historical_data)
        
        # Crear visualizaciones
        growth_fig = self.create_growth_comparison_chart(metrics_df)
        pie_chart = self.create_market_share_pie_chart(historical_data)
        ranking_chart = self.create_yearly_ranking_chart(historical_data)

        
        # Generar ranking y reporte
        ranking_df = self.create_historical_ranking(metrics_df)
        report_file = self.generate_historical_report(historical_data, metrics_df)
        
        # Mostrar resultados
        print("\n" + "="*30)
        print("ANÁLISIS HISTÓRICO COMPLETADO")
        print("="*30)
        
        print(f"\nPERÍODO ANALIZADO: {years} años")
        print(f"TECNOLOGÍAS ANALIZADAS: {len(metrics_df)}")
        
        print("\nTOP 5 TECNOLOGÍAS POR CRECIMIENTO HISTÓRICO (CAGR):")
        top_cagr = metrics_df.nlargest(5, 'cagr')[['technology', 'cagr']]
        for _, row in top_cagr.iterrows():
            print(f"  {row['technology']}: {row['cagr']:+.2f}% anual")
        
        print("\nTOP 5 TECNOLOGÍAS POR POPULARIDAD ACTUAL:")
        top_popular = metrics_df.nlargest(5, 'current_popularity')[['technology', 'current_popularity']]
        for _, row in top_popular.iterrows():
            print(f"  {row['technology']}: {row['current_popularity']:,.0f} menciones")
        
        print("\nRANKING HISTÓRICO COMPUESTO:")
        print(ranking_df.head(10).to_string(index=False, float_format='%.2f'))
        
        return historical_data, metrics_df, ranking_df, [growth_fig, pie_chart, ranking_chart]


def safe_get(url, params=None, headers=None, retries=3, wait_base=2):
    """
    Hace una solicitud HTTP con reintentos exponenciales para evitar bloqueos (403, 429).
    """
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r
            elif r.status_code in [429, 502, 503]:
                wait = (wait_base ** i) + random.uniform(0, 1)
                print(f"Rate limit ({r.status_code}), reintentando en {wait:.1f}s...")
                time.sleep(wait)
            else:
                r.raise_for_status()
        except Exception as e:
            print(f"Intento {i+1} fallido: {e}")
            time.sleep(wait_base ** i)
    raise Exception("Demasiados intentos fallidos o sin conexión.")

# Función principal 
def main():
    """
    Demostración del análisis histórico
    """
    analyzer = AnalizarTecnologias()
    
    # Ejecuta análisis histórico 
    historical_data, metrics_df, ranking_df, figures = analyzer.run_historical_analysis(
        years=3, 
        use_real_data=False
    )
    
    # Mostrar gráficos
    for fig in figures:
        fig.show()
    
    # Análisis adicional
    print("\n" + "="*30)
    print("RECOMENDACIONES ESTRATÉGICAS")
    print("="*30)
    
    # Tecnologías emergentes (alto crecimiento, popularidad media)
    emerging_mask = (metrics_df['cagr'] > metrics_df['cagr'].quantile(0.7)) & \
                   (metrics_df['current_popularity'] > metrics_df['current_popularity'].median())
    emerging_techs = metrics_df[emerging_mask].nlargest(3, 'cagr')
    
    # Tecnologías establecidas (alta popularidad, crecimiento estable)
    established_mask = (metrics_df['current_popularity'] > metrics_df['current_popularity'].quantile(0.7)) & \
                      (metrics_df['volatility'] < metrics_df['volatility'].median())
    established_techs = metrics_df[established_mask].nlargest(3, 'current_popularity')
    
    print("\n TECNOLOGÍAS EMERGENTES (Alto Crecimiento):")
    for _, tech in emerging_techs.iterrows():
        print(f"  • {tech['technology']} (CAGR: {tech['cagr']:+.2f}%)")
    
    print("\n TECNOLOGÍAS ESTABLECIDAS (Alta Popularidad):")
    for _, tech in established_techs.iterrows():
        print(f"  • {tech['technology']} (Popularidad: {tech['current_popularity']:,.0f})")
    
    print("\n TENDENCIAS DESTACADAS:")
    print(f"  • Crecimiento promedio del sector: {metrics_df['cagr'].mean():.2f}% anual")
    print(f"  • Tecnología más volátil: {metrics_df.loc[metrics_df['volatility'].idxmax(), 'technology']}")
    print(f"  • Tecnología más estable: {metrics_df.loc[metrics_df['volatility'].idxmin(), 'technology']}")

if __name__ == "__main__":
    main()