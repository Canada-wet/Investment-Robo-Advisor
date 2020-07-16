import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from pandas_datareader import data as web
from datetime import datetime as dt

# Set up global variables
stockpricedf = 0
financialreportingdf =0
discountrate=0.2
margin = 0.15


# Set up the app
app = dash.Dash(__name__)
server = app.server

app.css.append_css({
    "external_url":"https://codepen.io/chriddyp/pen/bWLwgP.css"
})

def portfolio_construction():
    dictlist=[]
    dictlist.append({'value':'Risk Parity', 'label':'Low Volatility (Risk Parity)'})
    dictlist.append({'value':'Max Sharpe Ratio', 'label':'Higher Returns (Max Sharpe Ratio)'})
    return dictlist

def test(inputvalue1, selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix):
    return "Hello "+inputvalue1[:-3] +', ' + "based on your information, your recommended strategy is:   " + selected_dropdown_value1 +' for Equity, ' +selected_dropdown_value2+' for Credit, and '+selected_dropdown_value3+' for PE. ' + 'And your suggested total portfolio construction method is :' + selected_dropdown_value_mix
test('Dan 40','Risk','Risk','Risk','Risk')


app.layout = html.Div([
    html.Div([
        html.H2("United All Weather Robo Advisor"),
        html.Img(src="/assets/UofT_logo.png")
    ], className="banner"),

    html.Div([
        html.Img(src="/assets/Robo_Advisor_logo.png",style={'height': '20%','width': '100%', 'display': 'inline-block'})
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div([
    html.Div([
        html.H2("Please input your name and age"),
        dcc.Input(id="risk-limit-input", value="Dan 40", type="text")
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div([
        html.H2("Are you risk taking?"),
        dcc.Input(id="risk_preference", value="No", type="text")
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div([
        html.H2("USD/CAD Exposure (please enter the proportion of USD)"),
        dcc.Input(id="USD_CAD exposure", value="0.5", type="text")
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),

    html.Div([
        html.H2("Cash Requirement for Regime or Tail Hedge (please enter a percentage)"),
        dcc.Input(id="cash_requirement", value="0.05", type="text")
        # html.Button(id="submit-button", n_clicks=0, children="Submit")
    ]),], style={'width': '60%', 'display': 'inline-block'}),

    html.Div([
        html.Img(src="/assets/Robo_bg.png",style={'height': '80%','width': '90%', 'display': 'inline-block'})
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div([

        html.H2('Sub Class Optimization'),
        # First let users choose stocks
        html.H3('Choose an algorithm for equity class'),
        dcc.Dropdown(
            id='my-dropdown1',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.H3('Choose an algorithm for credit class'),
        dcc.Dropdown(
            id='my-dropdown2',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.H3('Choose an algorithm for PE class'),
        dcc.Dropdown(
            id='my-dropdown3',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),

        html.P('')],style={'width': '40%', 'display': 'inline-block'}),
        html.Div([
        html.H2('Choose an algorithm for total portfolio'),
        dcc.Dropdown(
            id='mix-dropdown',
            #options=save_sp500_stocks_info()+save_self_stocks_info(),
            options=portfolio_construction(),
            value='Risk Parity'),
        html.P(''),
        # html.H2("Performance Analysis"),
        # html.Img(src="/assets/stock-icon.png",style={'height': '10%','width': '10%', 'display': 'inline-block'}),
        html.H3('Advice for you'),
        html.Table(id = 'my-table'),
        html.Button('Download Your Report ', id='submit-val', n_clicks=0)
        ], style={'width': '55%', 'float': 'right', 'display': 'inline-block'}),

        # html.Div([
        # html.H2('Risk Model'),
        #
        # html.H3("Stress Testing"),
        # html.Img(src="/assets/stock-icon.png",style={'height': '10%','width': '10%', 'display': 'inline-block'})
        # ], style={'width': '40%', 'display': 'inline-block'})

    ],style={'font-family':'PT Sans Narrow'})



# For the stocks graph
# @app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown_value):
#     global stockpricedf # Needed to modify global copy of stockpricedf
#     stockpricedf = web.DataReader(
#         selected_dropdown_value.strip(), data_source='yahoo',
#         start=dt(2013, 1, 1), end=dt.now())
#     return {
#         'data': [{
#             'x': stockpricedf.index,
#             'y': stockpricedf.Close
#         }]
#     }


# for the table
@app.callback(Output('my-table', 'children'),
            [Input('risk-limit-input','value'),
             Input('my-dropdown1', 'value'),
             Input('my-dropdown2','value'),
             Input('my-dropdown3','value'),
             Input('mix-dropdown','value')])
def generate_table(inputvalue1, selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix):
    # global financialreportingdf # Needed to modify global copy of financialreportingdf
    test_v = test(inputvalue1, selected_dropdown_value1,selected_dropdown_value2,selected_dropdown_value3,selected_dropdown_value_mix)

    # Header
    # return [html.Tr([html.Th(col) for col in financialreportingwritten.columns])] + [html.Tr([
    #     html.Td(financialreportingwritten.iloc[i][col]) for col in financialreportingwritten.columns
    # ]) for i in range(min(len(financialreportingwritten), max_rows))]
    return [html.Tr(html.Th(test_v))]




if __name__ == '__main__':
    app.run_server(debug=False)
