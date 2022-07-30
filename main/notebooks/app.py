#Import packages

import pandas as pd
import numpy as np
import os
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dash_table as dt


#Read in data for Github
law_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/TL-A243-2%20State%20Firearm%20Law%20Database%203.0.csv')
income_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/median_income_91-18.csv')
pop_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/states_and_pops.csv')
prez_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/prez_voting.csv')
own_df = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/FirearmsAnalysis/main/main/data/own_df.csv')

law_df = law_df.rename(
    columns={
        'Effective Date Year': 'Year',
        'State Postal Abbreviation':'ST'
    })
law_df['Year'] = law_df['Year'].astype(str)
law_type_choices = law_df['Law Class'].unique()

#Fix Mississippi and DC labels
law_df['State'] = np.where(law_df['State']=='MIssissippi', 'Mississippi', law_df['State'])
law_df['State'] = np.where(law_df['State']=='DIstrict of Columbia', 'District of Columbia', law_df['State'])


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
        dcc.Tab(label='2nd Tab',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
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
                        dcc.Graph(id='state_law_freq_map')
                    ],width=6),
                    dbc.Col([
                        dcc.Dropdown(
                            id='dropdown0',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in law_type_choices],
                            value=law_type_choices[0]
                        ),
                        dcc.Graph(id='law_types_timeline')
                    ],width=6)
                ])
            ]
        )

    ])
])



@app.callback(
    Output('state_law_freq_map','figure'),
    Input('radio1','value'))

def update_law_map(radio_select):

   
    if "All Laws" in radio_select:
        map_df = pd.DataFrame(law_df[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',

                    hover_name='State',
                    locationmode='USA-states',
                    labels={'Count':'# Laws Passed'},
                    scope='usa')
        return fig

    elif "Permissive" in radio_select:
        new_df = law_df[law_df['Effect']=="Permissive"]

        map_df = pd.DataFrame(new_df[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',

                    hover_name='State',
                    locationmode='USA-states',
                    labels={'Count':'# Laws Passed'},
                    scope='usa')
        return fig
    else:

        new_df = law_df[law_df['Effect']=="Restrictive"]

        map_df = pd.DataFrame(new_df[['State','ST']].value_counts()).reset_index()
        map_df = map_df.rename(columns={0: 'Count'})

        fig = px.choropleth(map_df,
                    locations='ST',
                    color='Count',
                    template='plotly_dark',

                    hover_name='State',
                    locationmode='USA-states',
                    labels={'Count':'# Laws Passed'},
                    scope='usa')
        return fig

@app.callback(
    Output('law_types_timeline', 'figure'), 
    Input('dropdown0', 'value') 
)
def timeline_graph(dd0):
    filtered = law_df[law_df['Law Class']==dd0]
    #filtered = law_df[law_df['Law Class']=="background checks"]

    test = filtered[['Year','Law Class']]
    test = filtered.groupby(['Year']).size().reset_index()
    test = test.rename(columns={0: 'Count'})



    fig = px.line(
        test, 
        x="Year", y="Count",
        #markers=True,
        #orientation='h',
        template='plotly_dark',
        #labels={'ones':'# Astronauts','country':'Country'},
        #title=f'# Astronauts by Country in {slider0}',
        #text_auto=True
    )

    
    return fig

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

#     elif tab == 'tab-3':
#         return html.Div([
#             html.H3('Tab content 3')
#         ])

#     elif tab == 'tab-4':
#         return html.Div([
#             html.H3('Tab content 4')
#         ])




# @app.callback(
#     Output("modal0", "is_open"),
#     Input("open0", "n_clicks"), 
#     Input("close0", "n_clicks"),
#     State("modal0", "is_open")
# )

# def toggle_modal0(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open



if __name__=='__main__':
	app.run_server()
