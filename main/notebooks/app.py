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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import warnings
from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
from sklearn.metrics.pairwise import cosine_similarity

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

#Tab #1 --> Welcome Page
        dcc.Tab(label='Welcome',value='tab-1',style=tab_style, selected_style=tab_selected_style,
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
                       html.P("1.) find relationships between state-level firearms legislation and the firearm-related homicide and suicides rates by state and year.",style={'color':'white'}),
                       html.P("2.) Understand the demographics of states with similar firearm-related homicide and suicide rates",style={'color':'white'}),


                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What data is being used for this analysis?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   
                   html.Div([
                       html.P(['1.) ',html.A('RAND State Firearm Law Database',href='https://www.rand.org/pubs/tools/TLA243-2-v2.html',style={'color':'white'})],style={'color':'white'}),
                       html.P(['2.) ',html.A('RAND State-Level Estimates of Household Firearm Ownership',href='https://www.rand.org/pubs/tools/TL354.html',style={'color':'white'})],style={'color':'white'}),

                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What are the limitations of this data?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("1.) Blah blah blah",style={'color':'white'}),
                   ])


               ]),
#Tab #2 --> State Level Legislation

        dcc.Tab(label='State-Level Legislation',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.P('Click a state on the map or select a law type:')
                    ],width=2),
                    dbc.Col([
                        dcc.RadioItems(
                                id='radio1',
                                options=[
                                    {'label': 'All Laws', 'value': 'All Laws'},
                                    {'label': 'Permissive', 'value': 'Permissive'},
                                    {'label': 'Restrictive', 'value': 'Restrictive'}
                                ],
                                value='All Laws',

                                labelStyle={'display': 'block','text-align': 'left'}

                        ),
                    ],width=4),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Law Descriptions", id="open1",color='secondary',style={"fontSize":18}),
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
                                        dbc.Button("Close", id="close1", className="ml-auto")
                                    ),
                                ],id="modal1",size="xl",scrollable=True),
                        ],className="d-grid gap-2"),
                    ],width=6),
                    dbc.Col([
                        dcc.Graph(id='state_law_freq_map')
                    ],width=6),
                    dbc.Col([
                        dcc.Graph(id='law_types_treemap'),
                        html.Br(),
                        html.Br(),
                        dcc.Graph(id='state_timeline')
                    ],width=6)
                ])
            ]
        ),
#Tab #3 --> Clustering

        dcc.Tab(label='Clustering',value='tab-3',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        dcc.RangeSlider(
                                    id='range_slider',
                                    min=1991,
                                    max=2020,
                                    step=1,
                                    value=[1991, 2020],
                                    allowCross=False,
                                    pushable=2,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                    ],width=6),
            
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown1',
                            options=[{'label': i, 'value': i} for i in cluster_choices],
                            value=cluster_choices[0],
                        )
                    ],width=6)
                ]),
                #Card row
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='card1')     
                    ],width=4),
                    dbc.Col([
                        dbc.Card(id='card2')     
                    ],width=4),
                    dbc.Col([
                        dbc.Card(id='card3')     
                    ],width=4)
                 ]),
                 dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='cluster_map')
                    ],width=12)
             
                ])
            ]
        ),
#Tab #4 --> Cosine Similarity - Comparing Laws

        dcc.Tab(label='Cosine Similarity',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown2',
                            options=[{'label': i, 'value': i} for i in law_type_choices],
                            value=law_type_choices[0],
                        )
                    ],width=12)

                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='cosine_matrix')
                    ])
                ]),
                dbc.Row([
                      dbc.Col([
                        dcc.Dropdown(
                            id='dropdown3',
                            options=[{'label': i, 'value': i} for i in law_list_choices],
                            value=law_list_choices[0],
                        )
                    ],width=6),
                      dbc.Col([
                        dcc.Dropdown(
                            id='dropdown4',
                            options=[{'label': i, 'value': i} for i in law_list_choices],
                            value=law_list_choices[0],
                        )
                    ],width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='card4'),
                        dbc.Card(id='card5')
                    ],width=6),
                    dbc.Col([
                        dbc.Card(id='card6'),
                        dbc.Card(id='card7')
                    ],width=6)
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


#------------------- TAB #2 -------------------#

#Configure reactivity for changing choropleth state map
@app.callback(
    Output('state_law_freq_map','figure'),
    Input('radio1','value')
) 

def update_law_map(radio_select):
   
    if "All Laws" in radio_select:
        test = law_df[law_df['Year']>=1991]
        map_df = pd.DataFrame(test[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',
                    hover_name='State',
                    locationmode='USA-states',
                    color_continuous_scale="Viridis",

                    labels={
                        'Count':'# Laws Passed',
                        'ST':'State'
                    },
                    title='All Laws by Type (1991-2020)',
                    scope='usa')
            
        return fig

    elif "Permissive" in radio_select:
        test = law_df[law_df['Year']>=1991]

        new_df = test[test['Effect']==radio_select]

        map_df = pd.DataFrame(new_df[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',
                    hover_name='State',
                    title='Permissive Laws by Type (1991-2020)',
                    color_continuous_scale="Viridis",

                    locationmode='USA-states',
                    labels={
                        'Count':'# Laws Passed',
                        'ST':'State'
                    },
                    scope='usa')
        return fig
    else:
        test = law_df[law_df['Year']>=1991]

        new_df = test[test['Effect']==radio_select]
        map_df = pd.DataFrame(new_df[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',
                    hover_name='State',
                    title='Restrictive Laws by Type (1991-2020)',
                    color_continuous_scale="Viridis",
                    locationmode='USA-states',
                    labels={
                        'Count':'# Laws Passed',
                        'ST':'State'
                    },
                    scope='usa')
        #fig.update_layout(colorscale='h', selector=dict(type='heatmap'))

        return fig

#Configure reactivity for treemap with inputs of law type and map clicking
@app.callback(
    Output('law_types_treemap', 'figure'), 
    Input('radio1','value'),
    Input('state_law_freq_map', 'clickData')
) 

def update_tree_map_on_click(radio_select,click_state):

#Condition 1/6: No Click, All Laws
    if not click_state and "All Laws" in radio_select:
        #raise dash.exceptions.PreventUpdate
        test = law_df[law_df['Year']>=1991]

        location = test['State'].sort_values().iloc[0]
        filtered = test[test['State']==location]
    
        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
        )
        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Laws Passed in {location}',
            color = 'Law Class'
        )
        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Condition 2/6: No Click, Permissive Laws

    elif not click_state and "Permissive" in radio_select:
        test = law_df[law_df['Year']>=1991]

        location = test['State'].sort_values().iloc[0]
        filtered = test[test['State']==location]
        filtered = filtered[filtered['Effect']==radio_select]

        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
        )
        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Laws Passed in {location}',
            color = 'Law Class'
        )
        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Condition 3/6: No Click, Restrictive Laws

    elif not click_state and "Restrictive" in radio_select:
        test = law_df[law_df['Year']>=1991]

        location = test['State'].sort_values().iloc[0]
        filtered = test[test['State']==location]
        filtered = filtered[filtered['Effect']==radio_select]

        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
        )
        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Laws Passed in {location}',
            color = 'Law Class'
        )
        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Condition 4/6: Yes Click, All Laws

    elif click_state and "All Laws" in radio_select:
        test = law_df[law_df['Year']>=1991]

        location = click_state['points'][0]['hovertext']
        filtered = test[test['State']==location]
    
        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
                )

        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Laws Passed in {location}',
            color = 'Law Class'
        )
        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Condition 5/6: Yes Click, Permissive Laws

    elif click_state and "Permissive" in radio_select:
        test = law_df[law_df['Year']>=1991]

        location = click_state['points'][0]['hovertext']
        filtered = test[test['State']==location]
        filtered = filtered[filtered['Effect']==radio_select]

        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
                )

        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Permissive Laws Passed in {location}',
            color = 'Law Class'
        )

        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Condition 6/6: Yes Click, Restrictive Laws

    else:
        test = law_df[law_df['Year']>=1991]

        location = click_state['points'][0]['hovertext']
        filtered = test[test['State']==location]
        filtered = filtered[filtered['Effect']==radio_select]

        class_df = pd.DataFrame(filtered['Law Class'].value_counts()).reset_index()[0:5]
        class_df = class_df.rename(
                    columns={
                        'index': 'Law Class',
                        'Law Class':'# Laws'
                    }
                )

        tree_fig = px.treemap(
            class_df, 
            path = ['Law Class'],
            values = '# Laws',
            template='plotly_dark',
            title=f'Top 5 Classes of Restrictive Laws Passed in {location}',
            color = 'Law Class'
        )
        tree_fig.update_traces(
            hovertemplate='# Laws=%{value}'
        )
        tree_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return tree_fig

#Configure reactivity for area chart based on click
@app.callback(
    Output('state_timeline', 'figure'), 
    Input('state_law_freq_map', 'clickData')
) 

def update_area_chart_on_click(click_state):
    if not click_state:
        #raise dash.exceptions.PreventUpdate
        location = law_df['State'].sort_values().iloc[0]
        filtered = law_df[law_df['State']==location]
        filtered = filtered[filtered['Year']>=1991]

        timeline_df = pd.DataFrame(filtered['Year'].value_counts()).reset_index()
        timeline_df = timeline_df.rename(
                    columns={
                        'index': 'Year',
                        'Year':'# Laws'
                    }
                )

        timeline_fig = px.area(
            timeline_df,
            x='Year',
            y='# Laws',
            template='plotly_dark',
            markers=True,
            title = f'Total # Laws Passed in {location} (1991 to 2020)'
        )
        timeline_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return timeline_fig

    else:
        location = click_state['points'][0]['hovertext']
        filtered = law_df[law_df['State']==location]
        filtered = filtered[filtered['Year']>=1991]


        timeline_df = pd.DataFrame(filtered['Year'].value_counts()).reset_index()
        timeline_df = timeline_df.rename(
                    columns={
                        'index': 'Year',
                        'Year':'# Laws'
                    }
        ).sort_values(by='Year',ascending=False)

        timeline_fig = px.area(
            timeline_df,
            x='Year',
            y='# Laws',
            template='plotly_dark',
            markers=True,
            title = f'Total # Laws Passed in {location} (1991 to 2020)'
 
        )
        timeline_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=200
        )
        return timeline_fig
#-----------------------------Tab #3: Clustering -----------------------------#

#Configure reactivity of cluster map controlled by range slider
@app.callback(
    Output('cluster_map', 'figure'), 
    Output('dropdown1', 'options'),
    Input('range_slider', 'value'),
    Input('dropdown1','value')
    

) 

def update_cluster_map(slider_range_values,dd1):#,state_choice):
    filtered = cluster_df[(cluster_df['Year']>=slider_range_values[0]) & (cluster_df['Year']<=slider_range_values[1])]
    #filtered = cluster_df[(cluster_df['Year']>=1991) & (cluster_df['Year']<=2020)]
    

    X = filtered#[fixed_names]
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

    
    #This is the filtered list that gets populated in the dropdown box
    cluster_list = X['cluster'].unique().tolist()
    cluster_list.sort()
    label = 'Cluster '
    new_cluster_list = [label + x for x in cluster_list]

    sortedX = X.sort_values(by='cluster',ascending=True)
    sortedX['cluster'] = sortedX['cluster'].astype('str')
    sortedX['cluster_num'] = sortedX['cluster'].astype(int)

    def assign_color(value):
        float_value = float(value)
        if float_value == 1:
            return "rgba(108, 102, 137, 1)"
        elif float_value == 2:
            return "rgba(210, 215, 211, 1)"
        elif float_value == 3:
            return "rgba(238, 238, 238, 1)"
        elif float_value == 4:
            return "rgba(189, 195, 199, 1)"
        elif float_value == 5:
            return "rgba(236, 240, 241, 1)"
        elif float_value == 6:
            return "rgba(149, 165, 166, 1)"
        else:
            return "rgba(191, 191, 191, 1)"

    #sortedX.to_csv('test_X.csv', index=False)
    sortedX['color'] = sortedX.apply(lambda x: assign_color(x["cluster_num"]), axis = 1)
    sortedX['color'] = np.where(sortedX['cluster_num']==int(dd1[-1]),'rgba(46, 204, 113, 1)',sortedX['color'])
    
    fig = px.scatter(
        sortedX,
        x="SuicidesA3", 
        y="HomicidesA3", 
        hover_data = {
            "State":True,
            "Year":True,
            "SuicidesA3":True,
            "HomicidesA3":True,
            "cluster":True
        },
        labels={
            'SuicidesA3':'3 Year Avg Suicide Rate',
            'HomicidesA3':'3 Year Avg Homicide Rate',
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
    fig.update_xaxes(title_text='Suicide Rate (3 Yr Avg)')
    fig.update_yaxes(title_text='Homicide Rate (3 Yr Avg)')
   

            
    return fig, [{'label':i,'value':i} for i in new_cluster_list]
    

#-----------------------------Tab #4: Cosine Similarity Matrix for Law Types -----------------------------#

#Configure reactivity for initial filters (dropdown2 and rangeslider2) to filter law dropdowns
@app.callback(
    Output('dropdown3', 'options'),#-----Filters the Law ID options
    Output('dropdown3', 'value'),
    Input('dropdown2', 'value') #----- Select the Law type
)
def set_law_options(selected_law_type):
    return [{'label': i, 'value': i} for i in law_type_id_dict[selected_law_type]], law_type_id_dict[selected_law_type][0]

@app.callback(
    Output('dropdown4', 'options'),#-----Filters the Law ID options
    Output('dropdown4', 'value'),
    Input('dropdown2', 'value') #----- Select the Law type
)
def set_law_options2(selected_law_type):
    return [{'label': i, 'value': i} for i in law_type_id_dict[selected_law_type]], law_type_id_dict[selected_law_type][0]



@app.callback(
    Output('cosine_matrix', 'figure'), 
    Input('dropdown2','value')
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
        columns=count_vectorizer.get_feature_names()
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
        zmax=1

    )

    return fig
    
# #Configure reactivity of cards based on dynamic dropdown box
@app.callback(
    Output('card4','children'),
    Output('card5','children'),
    Output('card6','children'),
    Output('card7','children'),
    Input('dropdown3', 'value'),
    Input('dropdown4', 'value')
) 

def update_cards1(dd3,dd4):
    filtered = law_df[law_df['Law ID'].isin([dd3,dd4])]
    #filtered = law_df[law_df['Law ID'].isin(['AK1003','ME1005'])]

    #Add in cluster before v after metrics here
    filter_cluster = cluster_df[['Law ID','SuicidesB3','SuicidesA3','HomicidesB3','HomicidesA3']]
    filter_cluster = cluster_df[cluster_df['Law ID'].isin([dd3,dd4])]
    #filter_cluster = filter_cluster[filter_cluster['Law ID'].isin(['AK1003','ME1005'])]
   
    stats_df = pd.merge(
        filtered,
        filter_cluster,
        how='left',
        on=['Law ID']
    )



    law_effect1 = stats_df[stats_df['Law ID']==dd3]['Effect'].values[0]
    law_effect2 = stats_df[stats_df['Law ID']==dd4]['Effect'].values[0]
    
    # law_year1 = round(stats_df[stats_df['Law ID']==dd3]['Year'].values[0])
    # law_year2 = round(stats_df[stats_df['Law ID']==dd4]['Year'].values[0])
    
    law1_hom_before = stats_df[stats_df['Law ID']==dd3]['HomicidesB3'].values[0]
    law1_sui_before = stats_df[stats_df['Law ID']==dd3]['SuicidesB3'].values[0]
    law2_hom_before = stats_df[stats_df['Law ID']==dd4]['HomicidesB3'].values[0]
    law2_sui_before = stats_df[stats_df['Law ID']==dd4]['SuicidesB3'].values[0]

    law1_hom_after = stats_df[stats_df['Law ID']==dd3]['HomicidesA3'].values[0]
    law1_sui_after = stats_df[stats_df['Law ID']==dd3]['SuicidesA3'].values[0]
    law2_hom_after = stats_df[stats_df['Law ID']==dd4]['HomicidesA3'].values[0]
    law2_sui_after = stats_df[stats_df['Law ID']==dd4]['SuicidesA3'].values[0]

    law1_hom_change = round(((law1_hom_before - law1_hom_after)/law1_hom_before)*100,2)
    law1_sui_change = round(((law1_sui_before - law1_sui_after)/law1_sui_before)*100,2)

    law2_hom_change = round(((law2_hom_before - law2_hom_after)/law2_hom_before)*100,2)
    law2_sui_change = round(((law2_sui_before - law2_sui_after)/law2_sui_before)*100,2)

    card4 = dbc.Card([
        dbc.CardBody([
            html.P(f'Law #{dd3} passed in '),
            html.H6(f'{law_effect1}'),
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card5 = dbc.Card([
        dbc.CardBody([
            html.P(f'Suicide Rate ∆: {law1_sui_change}%'),
            html.P(f'Homicide Rate ∆: {law1_hom_change}%'),
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    card6 = dbc.Card([
        dbc.CardBody([
            html.P(f'Law #{dd4} passed in '),
            html.H6(f'{law_effect2}'),
        ])
    ],
    style={'display': 'inline-block',
           #'width': '20%',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)


    card7 = dbc.Card([
        dbc.CardBody([
            html.P(f'Suicide Rate ∆: {law2_sui_change}%'),
            html.P(f'Homicide Rate ∆: {law2_hom_change}%')
        ])
    ],
    style={'display': 'inline-block',
           'text-align': 'center',
           'background-color': '#70747c',
           'color':'white',
           'fontWeight': 'bold',
           'fontSize':20},
    outline=True)

    return card4, card5, card6, card7
    

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



if __name__=='__main__':
	app.run_server()
