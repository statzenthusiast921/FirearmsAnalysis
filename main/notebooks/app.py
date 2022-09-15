#Import packages
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import warnings
from kneed import KneeLocator
from dash import dash_table as dt
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#from dash import dash_table as dt


#Read in data for Github
law_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/law_df_updated.csv')
cluster_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/cluster_df_updated.csv')


#Clean up the law dataset as needed
law_df = law_df.rename(
    columns={
        'Effective Date Year': 'Year',
        'State Postal Abbreviation':'ST'
    })
law_df = law_df[law_df['Effective Date'].notnull()]


law_type_choices = law_df['Law Class'].unique()
law_list_choices = law_df['Law ID'].unique()


#Fix Mississippi and DC labels
law_df['State'] = np.where(law_df['State']=='MIssissippi', 'Mississippi', law_df['State'])
law_df['State'] = np.where(law_df['State']=='DIstrict of Columbia', 'District of Columbia', law_df['State'])
#Filter to only 1991 laws and alter
law_df = law_df[law_df['Year']>=1991]


#-----Identify laws that have been repealed-----#

repealed_list = law_df.loc[law_df['Content'].str.contains("repealed", case=False)]
repealed_list = repealed_list[['Law ID']].reset_index()
del repealed_list['index']

keys = list(repealed_list.columns.values)
i1 = law_df.set_index(keys).index
i2 = repealed_list.set_index(keys).index
law_df = law_df[~i1.isin(i2)]

#Law Type --> Law ID Dictionary
law_type_id_dict = law_df.groupby('Law Class')['Law ID'].apply(list).to_dict()

#Clean up cluster dataset as needed
cluster_df.drop(cluster_df[cluster_df['SuicidesB3'] == 0].index, inplace=True)
cluster_df.drop(cluster_df[cluster_df['SuicidesA3'] == 0].index, inplace=True)
cluster_df.drop(cluster_df[cluster_df['HomicidesB3'] == 0].index, inplace=True)
cluster_df.drop(cluster_df[cluster_df['HomicidesA3'] == 0].index, inplace=True)

cluster_choices = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4"]

# Year --> State Dictionary
df_for_dict = cluster_df[['Year','State']]
df_for_dict = df_for_dict.drop_duplicates(subset='State',keep='first')
year_state_dict = df_for_dict.groupby('Year')['State'].apply(list).to_dict()
year_state_dict_sorted = {l: sorted(m) for l, m in year_state_dict.items()} #sort value by list

# Defining a function to visualise n-grams
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'color':'white',
    'backgroundColor': '#222222'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#626ffb',
    'color': 'white',
    'padding': '6px'
}



app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
#-------------------------------------------------------------------------------------#
#Tab #1 --> Welcome Page
        dcc.Tab(label='What is this project about?',value='tab-1',style=tab_style, selected_style=tab_selected_style,
               children=[
                   html.Div([
                       html.H1(dcc.Markdown('''**Firearms Analysis Dashboard**''')),
                       html.Br()
                   ]),
                   
                   html.Div([
                        html.P(dcc.Markdown('''**What is the purpose of this dashboard?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("This dashboard was created as a tool to: ",style={'color':'white'}),
                       html.P("1.) Explore the similarity in the text of the laws and determine if any regional relationships exist",style={'color':'white'}),
                       html.P("2.) Discover patterns in the text and determine the relative importance of certain words.",style={'color':'white'}),
                       html.P("3.) Determine if any latent groupings of laws exist when including regional, demographic data as well as sentiment data.",style={'color':'white'}),
                       html.P("4.) [Verb] A fourth thing.",style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What data is being used for this analysis?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   
                   html.Div([
                       html.P(['1.) ',html.A('RAND State Firearm Law Database',href='https://www.rand.org/pubs/tools/TLA243-2-v2.html')],style={'color':'white'}),
                       html.P(['2.) ',html.A('RAND State-Level Estimates of Household Firearm Ownership',href='https://www.rand.org/pubs/tools/TL354.html')],style={'color':'white'}),
                       html.P(['3.) Scraped ',html.A('firearm related deaths data by state and year',href='https://www.statefirearmlaws.org/states/')],style={'color':'white'}),
                       html.P(['4.) Scraped ',html.A('median income',href='https://fred.stlouisfed.org/release/tables?rid=118&eid=259194'),' data from the Federal Reserve Bank of St. Louis Economic Data website'],style={'color':'white'}),
                       html.P(['5.) Scraped ',html.A('population',href='https://fred.stlouisfed.org/release/tables?rid=118&eid=259194')," data from the Federal Reserve Bank of St. Louis Economic Data website"],style={'color':'white'}),
                       html.P(['5.) ',html.A('County-level voting histories',href='https://github.com/statzenthusiast921/US_Elections_Project/blob/main/Data/FullElectionsData.xlsx')," data that I compiled for a previous project"],style={'color':'white'}),

                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What are the limitations of this data?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("1.) Comprehensive data for every measure used was not always available for every state or every year needed in this analysis.  Therefore, I only included data that covered the range of years 1991 through 2020.",style={'color':'white'}),
                       html.P("2.) The District of Colubmia was not included in the analysis of firearm-related homicide and suicide data due to the challenge of finding enough data covering all 30 years.",style={'color':'white'}),
                       html.P("3.) ",style={'color':'white'})
                   ])
               ]),
#-------------------------------------------------------------------------------------#
#Tab #2 --> Exploratory Analysis
        dcc.Tab(label='Exploratory Analysis',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open1",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Instructions"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Below is a cosine similarity matrix coupled with a map of the United States.  These two plots can be updated by selecting a different law type.'),
                                            html.P('The matrix pairs each of the laws for the selected law type against each other and calculates a score measuring the similarity of the text.  The interpretation of this matrix is very similar to that of a correlation matrix as higher scores indicate more textual similarities vs. lower scores which indicate fewer textual similarities.'),
                                            html.P('The map indicates the # of laws of the selected law type passed between the years 1991 and 2020.')
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close1", className="ml-auto")
                                    ),
                            ],id="modal1",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=6),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Analysis", id="open2",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Analysis"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Using the scale to the side of the matrix, you can find laws that have similar textual usage by finding the higher cosine similarity scores.'),
                                            html.P('Generally, the highest scores within a law type are slight variations of other laws from the same state passed around the same time period.'),
                                            html.P('For most of the Waiting Period and Background Check laws, the text used contains almost identical language and results in perfect cosine similarity scores indicating there is not much of a regional variation in how these types of laws are written.')
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close2", className="ml-auto")
                                    ),
                            ],id="modal2",size="md",scrollable=True),
                        ],className="d-grid gap-2")

                    ],width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose Law Type:**'''),style={'color':'white'}),                        
                        dcc.Dropdown(
                            id='dropdown1',
                            options=[{'label': i, 'value': i} for i in law_type_choices],
                            value=law_type_choices[0],
                        )
                    ],width=6),
                    dbc.Col([
                        #Figure out a better solution here!
                        html.Label(dcc.Markdown('''**Invisible Text**'''),style={'color':'black'}),                        
                        html.Div([
                            dbc.Button("Click Here for Law Descriptions", id="open3",color='secondary',style={"fontSize":18}),
                                dbc.Modal([
                                    dbc.ModalHeader("Descriptions"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P(dcc.Markdown('''**1.) Carrying a concealed weapon (CCW)**''')),
                                            html.P('The act or practice of carrying a concealed firearm in public or the legal right to do so.'),
                                            html.P(dcc.Markdown('''**2.) Castle Doctrine**''')),
                                            html.P("Allows individuals to use deadly force in defense of themselves, others, and their homes without the duty to retreat"),
                                            html.P(dcc.Markdown('''**3.) Dealer License**''')),
                                            html.P('Requires dealers of firearms to be licensed by the state'),
                                            html.P(dcc.Markdown('''**4.) Minimum Age**''')),
                                            html.P('Establishes a minimum age for possession of a gun '),
                                            html.P(dcc.Markdown('''**5.) Registration**''')),
                                            html.P('Requires a recordkeeping system controlled by a government agency that stores the names of current owners of each firearm of a specific class and requires that these records are updated after firearms are transferred to a new owner (with few exceptions)'),
                                            html.P(dcc.Markdown('''**6.) Waiting Period**''')),
                                            html.P('Establishes the minimum amount of time sellers must wait before delivering a gun to a purchaser'),
                                            html.P(dcc.Markdown('''**7.) Prohibited Possessor**''')),
                                            html.P('Prohibits the possession of firearms by individuals adjudicated as being mentally incompetent, incapacitated, or disabled; '),
                                            html.P(dcc.Markdown('''**8.) Background Checks**''')),
                                            html.P('Requires a background check of individuals purchasing guns'),
                                            html.P(dcc.Markdown('''**9.) Local laws preempted by state**''')),
                                            html.P('Prohibits local laws'),
                                            html.P(dcc.Markdown('''**10.) Firearm removal at scene of domestic violence**''')),
                                            html.P('Requires police officers to seize a firearm at the scene of a domestic violence incident'),
                                            html.P(dcc.Markdown('''**11.) Firearms in college/university**''')),
                                            html.P('Prohibits the possession of all firearms on the property of all private colleges and universities'),
                                            html.P(dcc.Markdown('''**12.) Child Access Laws**''')),
                                            html.P('Mandates safe storage of guns and prohibits individuals from furnishing guns to minors'),
                                            html.P(dcc.Markdown('''**13.) Firearm Sales Restrictions**''')),
                                            html.P('Bans the sale of specific firearms'),
                                            html.P(dcc.Markdown('''**14.) Open Carry**''')),
                                            html.P('Dictates whether licenses for the open carrying of guns is required'),
                                            html.P(dcc.Markdown('''**15.) Safety Training Required**''')),
                                            html.P('Requires a safety training certificate for the purchase of a firearm'),
                                            html.P(dcc.Markdown('''**16.) Permit to Purchase**''')),
                                            html.P('Requires prospective purchasers to first obtain a license or permit from law enforcement'),
                                            html.P(dcc.Markdown('''**17.) Required reporting of lost or stolen firearms**''')),
                                            html.P('Requires victims of theft and loss of firearms to report the incident to relevant authorities')
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close3", className="ml-auto")
                                    ),
                                ],id="modal3",size="xl",scrollable=True),
                        ],className="d-grid gap-2"),

                    ])
                   
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='cosine_matrix')
                    ],width=6),
                    dbc.Col([
                        dcc.Graph(id='law_map')
                    ],width=6)
                ])                
            ]
        ),
#-------------------------------------------------------------------------------------#
#Tab #3 --> Patterns in Text --> N Grams, TF-IDF
        dcc.Tab(label='Patterns in Text',value='tab-3',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open4",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Instructions"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Blah blah blah')
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close4", className="ml-auto")
                                    ),
                            ],id="modal4",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=6),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Analysis", id="open5",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Analysis"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Blah blah blah'),
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close5", className="ml-auto")
                                    ),
                            ],id="modal5",size="md",scrollable=True),
                        ],className="d-grid gap-2")

                    ],width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose Law Type:**'''),style={'color':'white'}),                        
                        dcc.Dropdown(
                            id='dropdown2',
                            options=[{'label': i, 'value': i} for i in law_type_choices],
                            value=law_type_choices[0],
                        ),
                        html.Label(dcc.Markdown('''**Choose # of words:**'''),style={'color':'white'}),                        
                        dcc.Slider(
                            id='num_words_slider',
                            min=1,max=3,step=1,value=1,
                            marks={
                                1: '1',
                                2: '2',
                                3: '3'
                            }
                        )
                    ],width=6),
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose Laws to Compare:**'''),style={'color':'white'}),                        
                        dcc.Dropdown(
                            id='dropdown3',
                            options=[{'label': i, 'value': i} for i in law_list_choices],
                            value=law_list_choices[0],
                        ),
                        dcc.Dropdown(
                            id='dropdown4',
                            options=[{'label': i, 'value': i} for i in law_list_choices],
                            value=law_list_choices[0],
                        )
                    ],width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='n_gram_chart')
                    ],width=6),
                    dbc.Col([
                        dcc.Graph(id='tf_idf'),
                    ],width=6)
                ])
            ]
        ),
#-------------------------------------------------------------------------------------#
#Tab #4 --> Clustering
        dcc.Tab(label='Clustering',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open6",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Instructions"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Below is a scatter plot displaying the results from a cluster analysis with KPI metrics to help describe the cluster.  Each dot on the scatter plot represents a law.  Use the year range slider to filter the laws by year passed.'),
                                            html.P('To view metrics for a particular cluster or to better visualize the cluster on the graph, use the dropdown menu to select your desired cluster.  The laws that fall into this cluster will highlight green and every other law will remain grey.')

                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close6", className="ml-auto")
                                    ),
                            ],id="modal6",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=6),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Analysis", id="open7",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Analysis"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Depending on how the year range filter is utilized, the application of clusters is defined a little differently for some combination of years.'),
                                            html.P('However, there is a consistent trend for all variations of filters that the laws that fall into clusters with higher median population, higher median income, and generally higher median DEM vote share have lower suicide and homicide rates 3 years after the passing of the law.  Further, laws that fall into clusters with lower median population, lower median income, and generally higher median GOP vote share tend to have higher suicide rates, but not necessarily higher homicide rates 3 years after the passing of the law.')
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close7", className="ml-auto")
                                    ),
                            ],id="modal7",size="md",scrollable=True),
                        ],className="d-grid gap-2")

                    ],width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose a range of years:**'''),style={'color':'white'}),                        
                        dcc.RangeSlider(
                                    id='range_slider',
                                    min=1991,
                                    max=2020,
                                    marks=None,
                                    step=1,
                                    value=[1991, 2020],
                                    allowCross=False,
                                    pushable=2,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                    ],width=6),
            
                    dbc.Col([
                        html.Label(dcc.Markdown('''**Choose a cluster:**'''),style={'color':'white'}),                        
                        dcc.Dropdown(
                            id='dropdown5',
                            options=[{'label': i, 'value': i} for i in cluster_choices],
                            value=cluster_choices[0],
                        )
                    ],width=6)
                ]),
                #Card row
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='card1')     
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card2')     
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card3a')     
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card3b')     
                    ],width=3)
                 ]),
                 dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='cluster_map')
                    ],width=12)
             
                ])
            ]
        ),
#-------------------------------------------------------------------------------------#
#Tab #5 --> Predicting Suicide and Homicide

        dcc.Tab(label='Prediction',value='tab-5',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.P('Gamma model is the way to go')
                    ])
                ])
            ]
        )
    ])
])

#Configure Reactivity for Tab Colors
@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])

    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Tab content 5')
        ])


#-----------------------------Tab #2: Exploratory Analysis -----------------------------#

#Configure reactivity for cosine similarity matrix - law text
@app.callback(
    Output('cosine_matrix', 'figure'), 
    Input('dropdown1','value')
) 

def update_matrix(dd2):#,state_choice):
    filtered = law_df[law_df['Law Class']==dd2]
    filtered = filtered[filtered['ST']!="DC"]

    #Step 1: Take content column and convert to a list
    lg_list = filtered['Content_cleaned'].tolist()

    #Step 2: Create the Document Term Matrix
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(lg_list)

    doc_term_matrix = sparse_matrix.todense()
    df_test = pd.DataFrame(
        doc_term_matrix, 
        columns=count_vectorizer.get_feature_names_out()
    )

    #Step 3: Set up matrix
    array = cosine_similarity(df_test, df_test)
    matrix = pd.DataFrame(array,columns=filtered['Law ID'].tolist()) 
    
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    triangle = matrix.mask(mask)

    fig = px.imshow(
        triangle, 
        text_auto=True, 
        aspect="auto",
        x=triangle.columns,
        y=triangle.columns,
        template='plotly_dark',
        color_continuous_scale="Viridis",
        labels={
            'color':'Cosine Similarity'
        },
        zmin=0,
        zmax=1,
        title = f'Cosine Similarity Matrix: Text of {dd2.title()} Laws'

    )

    return fig

#Configure reactivity for choropleth map showing which states have passed certain types of laws
@app.callback(
    Output('law_map', 'figure'), 
    Input('dropdown1','value')
)
def law_map_function(dd2):
    filtered = law_df[law_df['Law Class']==dd2]
    filtered = filtered[filtered['ST']!="DC"]

    filtered = filtered[filtered['Year']>=1991]
    map_df = pd.DataFrame(filtered[['State','ST']].value_counts()).reset_index()
    map_df = map_df.rename(columns={0: 'Count'})

    fig = px.choropleth(map_df,
                            locations='ST',
                            color='Count',
                            template='plotly_dark',
                            hover_name='State',
                            locationmode='USA-states',
                            color_continuous_scale="Viridis",
                            title = f'# {dd2.title()} Laws Passed by State (1991-2020)',
                            labels={
                                    'Count':'# Laws Passed',
                                    'ST':'State'
                            },
                            scope='usa')
  
    fig.update_layout(coloraxis={"colorbar":{"dtick":1}})

    return fig

#----------------------------- Tab #3: Patterns in Text -----------------------------#

@app.callback(
    Output('dropdown3', 'options'), #--> filter law options
    Output('dropdown3', 'value'),
    Input('dropdown2', 'value') #--> choose law type
)
def set_law_options1(selected_law):
    return [{'label': i, 'value': i} for i in law_type_id_dict[selected_law]], law_type_id_dict[selected_law][0],

@app.callback(
    Output('dropdown4', 'options'), #--> filter law options
    Output('dropdown4', 'value'),
    Input('dropdown2', 'value') #--> choose law type
)
def set_law_options2(selected_law):
    return [{'label': i, 'value': i} for i in law_type_id_dict[selected_law]], law_type_id_dict[selected_law][1]

#Configure reactivity of n-gram chart
@app.callback(
    Output('n_gram_chart','figure'),
    Input('num_words_slider','value'),
    Input('dropdown2','value')
)

def update_n_gram_chart(word_num_slider,dd3):
    filtered = law_df[law_df['Law Class']==dd3]
    
    word_num = word_num_slider
    top_ngrams = get_top_ngram(filtered['Content_cleaned'],word_num)[:10]
    top_words = pd.DataFrame(top_ngrams,columns=['word','count'])
    top_words = top_words.sort_values('count',ascending=True)

    bar_fig = px.bar(
        top_words, 
        x='count', 
        y='word',
        orientation='h',
        title=f'Words Most Commonly Used in {dd3.title()} Laws',
        labels={'count':'Frequency'},
        template='plotly_dark'
    )

    bar_fig.update_layout(
        coloraxis_showscale=False, 
        yaxis_title=None,
        margin=dict(l=20, r=20, t=45, b=20),
        title={
            'xanchor':'center',
            'x':0.5
        }
    )
    return bar_fig


#Configure reactivity for tf_idf bar chart
@app.callback(
    Output('tf_idf', 'figure'), 
    Input('dropdown2','value'),
    Input('dropdown3','value'),
    Input('dropdown4','value')

) 

def update_tf_idf_bar_chart(dd3,dd4,dd5):
    filtered = law_df[law_df['Law Class']==dd3]
    law_ids = pd.DataFrame(filtered['Law ID'])

    #Create a list of all the laws
    corpus = filtered['Content_cleaned'].tolist()

    tr_idf_model  = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(corpus)  

    tf_idf_array = tf_idf_vector.toarray()
    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
    df_tf_idf['Law ID'] = law_ids['Law ID'].values

    law1 = pd.DataFrame(df_tf_idf[df_tf_idf['Law ID']==dd4])
    law2 = pd.DataFrame(df_tf_idf[df_tf_idf['Law ID']==dd5])

    del law1['Law ID']
    del law2['Law ID']


    #Let's create two lists with column headers and another with values
    words1 = law1.columns.values.tolist()
    words2 = law2.columns.values.tolist()

    values1 = law1[words1].values.tolist()[0]
    values2 = law2[words2].values.tolist()[0]

    law1_df = pd.DataFrame([words1,values1]).T
    law2_df = pd.DataFrame([words2,values2]).T

    #Drop last observation --> result of transposing unwanted
    law1_df.drop(law1_df.tail(1).index,inplace=True)
    law2_df.drop(law2_df.tail(1).index,inplace=True)

    #Reframe tf-idf dfs
    law1_df.rename(
        columns={
            0: 'word',
            1: 'tfidf'
        },inplace=True
    )
    law2_df.rename(
        columns={
            0: 'word',
            1: 'tfidf'
        },inplace=True
    )

    law1_words = law1_df.sort_values(by='tfidf',ascending=False)
    law2_words = law2_df.sort_values(by='tfidf',ascending=False)

    #Get the top 10 words sorted by tfidf
    law1_words = law1_words.head(10)
    law2_words = law2_words.head(10)
    
    #Plot the two graphs next to each other
    fig = make_subplots(rows=1, cols=2)

    bar_fig1 = px.bar(law1_words, x='word', y='tfidf')
    bar_fig2 = px.bar(law2_words, x='word', y='tfidf')

    bar_fig1.update_traces(marker_color='#D7504D')
    bar_fig2.update_traces(marker_color='#D7504D')

    figures = [bar_fig1, bar_fig2]

    fig = make_subplots(rows=len(figures), cols=1) 

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            fig.append_trace(figure["data"][trace], row=i+1, col=1)
            fig.update_layout(
                template='plotly_dark',
                title=f'TF-IDF Scores for {dd3.title()} Laws',
                margin=dict(l=20, r=20, t=45, b=20)

            )
            
    return fig
    
#----------------------------- Tab #4: Clustering -----------------------------#


#Configure reactivity of cluster map controlled by range slider
@app.callback(
    Output('cluster_map', 'figure'), 
    Output('dropdown5', 'options'),
    Output('card1','children'),
    Output('card2','children'),
    Output('card3a','children'),
    Output('card3b','children'),
    Input('range_slider', 'value'),
    Input('dropdown5','value')
    
) 

def update_cluster_map(slider_range_values,dd1):#,state_choice):
    filtered = cluster_df[(cluster_df['Year']>=slider_range_values[0]) & (cluster_df['Year']<=slider_range_values[1])]
    #filtered = cluster_df[(cluster_df['Year']>=1991) & (cluster_df['Year']<=2020)]
    filtered = filtered.reset_index()
    index_df = filtered[['index','Law ID']]
    X = filtered.copy()#[fixed_names]

    del X['Law ID'], X['UniqueID'], X['ST'], X['Suicides'], X['Homicides'], X['HomicidesB3'], X['SuicidesB3']

    #Step 2.) Imputation needed
    states = pd.DataFrame(X[['State']])
    not_states = X.loc[:, ~X.columns.isin(['State'])]

    #Step 2a.) Impute the non-text columns
    imputer = KNNImputer(n_neighbors=5)
    not_states_fixed = pd.DataFrame(imputer.fit_transform(not_states),columns=not_states.columns)

    #Step 3.) Perform clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(not_states_fixed)

    #Defining the kmeans function with initialization as k-means++
    kmeans = KMeans(n_clusters=6, init='k-means++',random_state=42)

    #Fitting the k means algorithm on scaled data
    kmeans.fit(data_scaled)

    SSE = []
    for cluster in range(1,20):
        kmeans = KMeans(n_clusters = cluster, init='k-means++',random_state=42)
        kmeans.fit(data_scaled)
        SSE.append(kmeans.inertia_)
        
    kl = KneeLocator(
        range(1, 20), SSE, curve="convex", direction="decreasing"
    )

    elbow = kl.elbow

    kmeans = KMeans(n_clusters = elbow, init='k-means++',random_state=42)
    kmeans.fit(data_scaled)
    pred = kmeans.predict(data_scaled)

    frame = pd.DataFrame(data_scaled)
    frame['cluster'] = pred
    frame['cluster'].value_counts()

    clusters = frame['cluster'] +1

    not_states_fixed = not_states_fixed.dropna()
    not_states_fixed['cluster'] = clusters.values
    not_states_fixed['cluster'] = not_states_fixed['cluster'].astype('str')

    state_list = states['State'].values.tolist()

    not_states_fixed['State'] = state_list

    X = not_states_fixed

    #Add the Law ID back into df
    X = pd.merge(
        X,
        index_df,
        how='inner',
        on=['index']
    )

    #This is the filtered list that gets populated in the dropdown box
    cluster_list = X['cluster'].unique().tolist()
    cluster_list.sort()
    label = 'Cluster '
    new_cluster_list = [label + x for x in cluster_list]

    sortedX = X.sort_values(by='cluster',ascending=True)
    sortedX['cluster'] = sortedX['cluster'].astype('str')
    sortedX['cluster_num'] = sortedX['cluster'].astype(int)

    sortedX['color'] = np.where(sortedX['cluster_num']==int(dd1[-1]),'rgba(46, 204, 113, 1)','rgba(108, 102, 137, 1)')
    
    fig = px.scatter(
        sortedX,
        x="SuicidesA3", 
        y="HomicidesA3", 
        hover_name = 'Law ID',
        hover_data = {
            "State":True,
            "Year":True,
            "SuicidesA3":True,
            "HomicidesA3":True,
            "cluster":True
        },
        labels={
            'SuicidesA3':'Avg Suicide Rate',
            'HomicidesA3':'Avg Homicide Rate',
            'cluster':'Cluster'
        },             
        template='plotly_dark',
    )
    fig.update_traces(
        marker=dict(
            color = sortedX["color"],
            size = 10,
            line=dict(width=0.5,color='white')),
        selector=dict(mode='markers')
    )
    fig.update_layout(showlegend=True)

 
    fig.update_xaxes(title_text='Suicide Rate (3 Yr Avg) Post Law Passing')
    fig.update_yaxes(title_text='Homicide Rate (3 Yr Avg) Post Law Passing')
    
    #Filter for cards
    sortedX['cluster_col'] = "Cluster " + sortedX['cluster'] 
    card_filter = sortedX[sortedX['cluster_col']==dd1]

    stat1 = round(card_filter['Pop'].median())
    stat2 = round(card_filter['Income'].median())
    stat3a = round(card_filter['DEM_Perc'].median()*100,1)
    stat3b = round(card_filter['GOP_Perc'].median()*100,1)

    card1 = dbc.Card([
        dbc.CardBody([
            html.P('Median Population'),
            html.H6(f'{stat1:,.0f}'),
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card2 = dbc.Card([
        dbc.CardBody([
            html.P('Median Income'),
            html.H6(f'${stat2:,.0f}')
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card3a = dbc.Card([
        dbc.CardBody([
            html.P('Median DEM Vote %'),
            html.H6(f'{stat3a}%'),
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card3b = dbc.Card([
        dbc.CardBody([
            html.P('Median GOP Vote %'),
            html.H6(f'{stat3b}%'),
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    return fig, [{'label':i,'value':i} for i in new_cluster_list], card1, card2, card3a, card3b
    
#----------Configure reactivity for Button #1 (Instructions) --> Tab #2----------#
@app.callback(
    Output("modal1", "is_open"),
    Input("open1", "n_clicks"), 
    Input("close1", "n_clicks"),
    State("modal1", "is_open")
)

def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


#----------Configure reactivity for Button #2 (Analysis) --> Tab #2----------#
@app.callback(
    Output("modal2", "is_open"),
    Input("open2", "n_clicks"), 
    Input("close2", "n_clicks"),
    State("modal2", "is_open")
)

def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


#----------Configure reactivity for Button #3 (Law Descriptions) --> Tab #2----------#
@app.callback(
    Output("modal3", "is_open"),
    Input("open3", "n_clicks"), 
    Input("close3", "n_clicks"),
    State("modal3", "is_open")
)

def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


#----------Configure reactivity for Button #4 (Instructions) --> Tab #3----------#
@app.callback(
    Output("modal4", "is_open"),
    Input("open4", "n_clicks"), 
    Input("close4", "n_clicks"),
    State("modal4", "is_open")
)

def toggle_modal4(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

#----------Configure reactivity for Button #5 (Analysis) --> Tab #3----------#
@app.callback(
    Output("modal5", "is_open"),
    Input("open5", "n_clicks"), 
    Input("close5", "n_clicks"),
    State("modal5", "is_open")
)

def toggle_modal5(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

#----------Configure reactivity for Button #6 (Instructions) --> Tab #4----------#
@app.callback(
    Output("modal6", "is_open"),
    Input("open6", "n_clicks"), 
    Input("close6", "n_clicks"),
    State("modal6", "is_open")
)

def toggle_modal6(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

#----------Configure reactivity for Button #6 (Analysis) --> Tab #4----------#
@app.callback(
    Output("modal7", "is_open"),
    Input("open7", "n_clicks"), 
    Input("close7", "n_clicks"),
    State("modal7", "is_open")
)

def toggle_modal7(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__=='__main__':
	app.run_server()
