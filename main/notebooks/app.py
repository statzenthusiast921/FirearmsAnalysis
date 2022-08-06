#Import packages

import pandas as pd
import numpy as np
import os
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
#from dash import dash_table as dt


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
law_df = law_df[law_df['Effective Date'].notnull()]


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
                        html.P('Select Law Type:')
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
                                            html.P(dcc.Markdown('''**1.) carrying a concealed weapon (ccw)**''')),
                                            html.P('The act or practice of carrying a concealed firearm in public or the legal right to do so.'),
                                            html.P(dcc.Markdown('''**2.) castle doctrine**''')),
                                            html.P("The legal doctrine that designates a person's abode or any legally occupied place as a place in which that person has protections and immunities permitting one, in certain circumstances, to use force (up to and including deadly force) to defend oneself against an intruder, free from legal prosecution for the consequences of the force used."),
                                            html.P(dcc.Markdown('''**3.) dealer license**''')),
                                            html.P('Requires dealers of firearms to be licensed by the state'),
                                            html.P(dcc.Markdown('''**4.) minimum age**''')),
                                            html.P('Establishes a minimum age for possession of a gun '),
                                            html.P(dcc.Markdown('''**5.) registration**''')),
                                            html.P('Requires a recordkeeping system controlled by a government agency that stores the names of current owners of each firearm of a specific class and requires that these records are updated after firearms are transferred to a new owner (with few exceptions)'),
                                            html.P(dcc.Markdown('''**6.) waiting period**''')),
                                            html.P('Establishes the minimum amount of time sellers must wait before delivering a gun to a purchaser'),
                                            html.P(dcc.Markdown('''**7.) prohibited possessor**''')),
                                            html.P('Prohibits the possession of firearms by individuals adjudicated as being mentally incompetent, incapacitated, or disabled; '),
                                            html.P(dcc.Markdown('''**8.) background checks**''')),
                                            html.P('Requires a background check of individuals purchasing guns'),
                                            html.P(dcc.Markdown('''**9.) local laws preempted by state**''')),
                                            html.P('Prohibits local laws'),
                                            html.P(dcc.Markdown('''**10.) firearm removal at scene of domestic violence**''')),
                                            html.P('Requires police officers to seize a firearm at the scene of a domestic violence incident'),
                                            html.P(dcc.Markdown('''**11.) firearms in college/university**''')),
                                            html.P('Prohibits the possession of all firearms on the property of all private colleges and universities'),
                                            html.P(dcc.Markdown('''**12.) child access laws**''')),
                                            html.P('Mandates safe storage of guns and prohibits individuals from furnishing guns to minors'),
                                            html.P(dcc.Markdown('''**13.) firearm sales restrictions**''')),
                                            html.P('Bans the sale of specific firearms'),
                                            html.P(dcc.Markdown('''**14.) open carry**''')),
                                            html.P('Dictates whether licenses for the open carrying of guns is required'),
                                            html.P(dcc.Markdown('''**15.) safety training required**''')),
                                            html.P('Requires a safety training certificate for the purchase of a firearm'),
                                            html.P(dcc.Markdown('''**16.) permit to purchase**''')),
                                            html.P('Requires prospective purchasers to first obtain a license or permit from law enforcement'),
                                            html.P(dcc.Markdown('''**17.) required reporting of lost or stolen firearms**''')),
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

#     elif tab == 'tab-3':
#         return html.Div([
#             html.H3('Tab content 3')
#         ])

#     elif tab == 'tab-4':
#         return html.Div([
#             html.H3('Tab content 4')
#         ])


#------------------- TAB #2 -------------------#

#Configure reactivity for changing choropleth state map
@app.callback(
    Output('state_law_freq_map','figure'),
    Input('radio1','value')
) 

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
        new_df = law_df[law_df['Effect']==radio_select]

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

        new_df = law_df[law_df['Effect']==radio_select]

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
    Output('law_types_treemap', 'figure'), 
    Input('radio1','value'),
    Input('state_law_freq_map', 'clickData')
) 

def update_tree_map_on_click(radio_select,click_state):
    if not click_state:
        #raise dash.exceptions.PreventUpdate
        location = law_df['State'].sort_values().iloc[0]
        filtered = law_df[law_df['State']==location]
    

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

    if "All Laws" in radio_select:
        location = click_state['points'][0]['hovertext']
        filtered = law_df[law_df['State']==location]
    

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

    elif "Permissive" in radio_select:
        location = click_state['points'][0]['hovertext']
        filtered = law_df[law_df['State']==location]
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
    else:
        location = click_state['points'][0]['hovertext']
        filtered = law_df[law_df['State']==location]
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

#Configure reactivity for line chart based on click
@app.callback(
    Output('state_timeline', 'figure'), 
    Input('state_law_freq_map', 'clickData')
) 

def update_linechart_on_click(click_state):
    if not click_state:
        #raise dash.exceptions.PreventUpdate
        location = law_df['State'].sort_values().iloc[0]
        filtered = law_df[law_df['State']==location]
        filtered = filtered[filtered['Year']>=1991]

        #filtered = law_df[law_df['State']=="Missouri"]


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
