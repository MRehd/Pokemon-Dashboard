# Importing the necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Getting and parsing pokémon data
url = "https://gist.github.com/simsketch/1a029a8d7fca1e4c142cbfd043a68f19"
r = requests.get(url)
df = pd.read_html(r.text)
data = df[0]
del data[0]
data.columns = data.iloc[0]
data.drop(0, inplace=True)
data.reset_index(inplace=True)
del data["index"]

# Function to parse the dataframe
def tidy_df(dataframe):
    def split_data(data):
        result = []
        for column in data.split(","):
            result.append(column)
        return result

    new_columns = split_data(dataframe.columns.tolist()[0])
    Dataframe = pd.DataFrame(columns=new_columns)

    for row in dataframe.index.tolist():
        row_to_append = {}
        new_row_list = split_data(dataframe.iloc[row][0])
        for column in new_columns:
            column_index = new_columns.index(column)
            row_to_append[column] = new_row_list[column_index]
        Dataframe = Dataframe.append(row_to_append, ignore_index=True)

    return Dataframe


data = tidy_df(data)

# Requesting Pokémon Types Strengths and Weaknesses information
url3 = "https://github.com/zonination/pokemon-chart/blob/master/chart.csv"
r3 = requests.get(url3)
strengths = pd.read_html(r3.text)[0]
strengths = strengths.dropna(axis=1, how="all")

# Creating SCORE column
data["SCORE"] = np.nan

# Converting strings into floats
data[
    ["HEIGHT", "WEIGHT", "HP", "ATK", "DEF", "SP_ATK", "SP_DEF", "SPD", "TOTAL"]
] = data[
    ["HEIGHT", "WEIGHT", "HP", "ATK", "DEF", "SP_ATK", "SP_DEF", "SPD", "TOTAL"]
].astype(
    float
)

# Simulating the Pokémon fights
print("Please hold while the simulations are running. This might take a few minutes.")
for attacker in data.index:
    total_score = 0
    Attack_of_attacking_pokemon = data.loc[attacker, "ATK"]  #
    Sp_attack_of_attacking_pokemon = data.loc[attacker, "SP_ATK"]  #
    Attacker_type1 = data.loc[attacker, "TYPE1"]
    Attacker_type2 = data.loc[attacker, "TYPE2"]
    for defender in data.index:
        Defense_of_defending_pokemon = data.loc[defender, "DEF"]  #
        Sp_defense_of_defending_pokemon = data.loc[defender, "SP_DEF"]  #

        Defender_type1 = data.loc[defender, "TYPE1"]
        Defender_type2 = data.loc[defender, "TYPE2"]

        coef_1 = strengths.loc[
            strengths["Attacking"] == Attacker_type1, Defender_type1
        ].iloc[0]
        try:
            coef_2 = strengths.loc[
                strengths["Attacking"] == Attacker_type2, Defender_type1
            ].iloc[0]
        except:
            coef_2 = 0
        try:
            coef_3 = strengths.loc[
                strengths["Attacking"] == Attacker_type1, Defender_type2
            ].iloc[0]
        except:
            coef_3 = 0
        try:
            coef_4 = strengths.loc[
                strengths["Attacking"] == Attacker_type2, Defender_type2
            ].iloc[0]
        except:
            coef_4 = 0
        coef = max([coef_1, coef_2, coef_3, coef_4])

        Attack_of_attacking_pokemon_final = Attack_of_attacking_pokemon * coef
        Sp_attack_of_attacking_pokemon_final = Sp_attack_of_attacking_pokemon * coef
        score = (
            Attack_of_attacking_pokemon_final - Defense_of_defending_pokemon
        ) * 0.8 + (
            Sp_attack_of_attacking_pokemon_final - Sp_defense_of_defending_pokemon
        ) * 0.2
        total_score += score

    data.loc[attacker, "SCORE"] = total_score

# Normalizing the SCORE
norm = data["SCORE"].values
data["NORM_SCORE"] = MinMaxScaler().fit_transform(norm.reshape(-1, 1))

# Renaming Pokémons with the same name but different characteristics
for name in data["NAME"].tolist():
    if data[data["NAME"] == name].shape[0] > 1:
        data.loc[data["NAME"] == name, "NAME"] = (
            name + " " + data.loc[data["NAME"] == name, "CODE"]
        )

# Function for the Pokémon Type Ranking Graph
def type_ranking():

    pkm_types = {}

    for i in data["TYPE1"].unique().tolist():
        pkm_types[i] = data.loc[data["TYPE1"] == i, "NORM_SCORE"].mean()

    pkm_types_sort = sorted(pkm_types.items(), key=lambda x: x[1], reverse=False)
    pkm_types_sort_x = [round(x[1], 3) for x in pkm_types_sort]
    pkm_types_sort_y = [y[0] for y in pkm_types_sort]

    fig1 = px.bar(
        x=pkm_types_sort_x,
        y=pkm_types_sort_y,
        labels={"x": "Average Pokémon Type Score", "y": "Pokémon Type"},
        color=pkm_types_sort_x,
        color_continuous_scale=px.colors.sequential.Brwnyl,
        title="Ranking of Pokémon Types",
        orientation="h",
    )

    fig1.update_layout(
        title=dict(x=0.5), margin=dict(l=0, r=20, t=60, b=20), paper_bgcolor="#D6EAF8"
    )

    fig1.update_traces(texttemplate=pkm_types_sort_x)

    return fig1


# Function for the Pokémon Ranking by Type Graph
def pokemon_by_type_graph(Type):
    pkm_by_type = {}
    for i in (
        data.loc[data["TYPE1"] == Type, "NAME"].tolist()
        + data.loc[data["TYPE2"] == Type, "NAME"].tolist()
    ):
        pkm_by_type[i] = data.loc[data["NAME"] == i, "NORM_SCORE"].iloc[0]

    pkm_by_type_sort = sorted(pkm_by_type.items(), key=lambda x: x[1], reverse=False)
    pkm_by_type_sort_x = [round(x[1], 3) for x in pkm_by_type_sort]
    pkm_by_type_sort_y = [y[0] for y in pkm_by_type_sort]

    fig2 = px.bar(
        x=pkm_by_type_sort_x,
        y=pkm_by_type_sort_y,
        labels={"x": f"{Type} Pokémon Score", "y": "Pokémon Name"},
        color=pkm_by_type_sort_x,
        color_continuous_scale=px.colors.sequential.Brwnyl,
        title=f"{Type} Pokémon Rank",
        orientation="h",
        height=2500,
    )

    fig2.update_layout(
        title=dict(x=0.5), margin=dict(l=25, r=25, t=60, b=20), paper_bgcolor="#D6EAF8"
    )

    fig2.update_traces(texttemplate=pkm_by_type_sort_x)

    return fig2


# Function for the Top10 Pokémon Graph
def top_x_pokemon(f):
    pkm = (
        data.sort_values("NORM_SCORE", ascending=False)
        .head(f)
        .copy()
        .sort_values("NORM_SCORE", ascending=True)
    )

    fig3 = px.bar(
        pkm,
        x=pkm["NORM_SCORE"],
        y=pkm["NAME"],
        labels={"x": "Pokémon Score", "y": "Pokémon Name"},
        color=pkm["NORM_SCORE"].values,
        color_continuous_scale=px.colors.sequential.Brwnyl,
        title="Global Ranking of Pokémons",
        orientation="h",
    )

    fig3.update_layout(
        title=dict(x=0.5), margin=dict(l=25, r=25, t=60, b=20), paper_bgcolor="#D6EAF8"
    )

    fig3.update_traces(texttemplate=[round(x, 3) for x in pkm["NORM_SCORE"].tolist()])

    return fig3


# Function for the Attack vs Score graph
def atk_score():
    fig4 = px.scatter(
        data,
        x="ATK",
        y="NORM_SCORE",
        labels={"x": "Pokémon ATK", "y": "Pokémon Score"},
        color=data["TYPE1"].values,
        title="Pokémon ATK vs Score",
        hover_name="NAME",
    )
    fig4.update_layout(
        title=dict(x=0.5), margin=dict(l=25, r=25, t=60, b=20), paper_bgcolor="#D6EAF8"
    )
    return fig4


# Function for the Defense vs Score Graph
def def_score():
    fig5 = px.scatter(
        data,
        x="DEF",
        y="NORM_SCORE",
        labels={"x": "Pokémon DEF", "y": "Pokémon Score"},
        color=data["TYPE1"].values,
        title="Pokémon DEF vs Score",
        hover_name="NAME",
    )
    fig5.update_layout(
        title=dict(x=0.5), margin=dict(l=25, r=25, t=60, b=20), paper_bgcolor="#D6EAF8"
    )
    return fig5


# Function for the histogram plots
def make_hist(attribute):
    fig6 = px.histogram(data, x=attribute, title=f"Pokémon {attribute} Distribution")

    fig6.update_layout(
        title=dict(x=0.5), margin=dict(l=25, r=25, t=60, b=20), paper_bgcolor="#D6EAF8"
    )
    return fig6


# Building the Dash app
fig1 = type_ranking()
fig2 = top_x_pokemon(10)
fig4 = atk_score()
fig5 = def_score()
fig6 = make_hist("ATK")
fig7 = make_hist("NORM_SCORE")

app = dash.Dash()

app.layout = html.Div(
    [
        html.Img(
            src="https://www.pinclipart.com/picdir/big/379-3791327_pokemon-logos-png-vector-pokemon-logo-transparent-background.png"
        ),
        html.H2(
            "Pokémon Analytics",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        html.Div(
            children="Analysis of Pokémon Strengths and Types",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        html.Div(
            children=dcc.Graph(
                id="atk-score-graph",
                figure=fig4,
            ),
            style={"width": "50%", "display": "inline-block"},
        ),
        html.Div(
            children=dcc.Graph(
                id="def-score-graph",
                figure=fig5,
            ),
            style={"width": "50%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="dist-1-chart-menu",
                    options=[
                        {"label": i, "value": i}
                        for i in sorted(
                            [
                                "HEIGHT",
                                "WEIGHT",
                                "HP",
                                "ATK",
                                "DEF",
                                "SP_ATK",
                                "SP_DEF",
                                "NORM_SCORE",
                            ]
                        )
                    ],
                    value="ATK",
                    style={"width": "100%"},
                )
            ]
        ),
        dcc.Graph("dist-1-chart", style={"width": "100%", "display": "inline-block"}),
        html.Div(
            [
                dcc.Dropdown(
                    id="dist-2-chart-menu",
                    options=[
                        {"label": i, "value": i}
                        for i in sorted(
                            [
                                "HEIGHT",
                                "WEIGHT",
                                "HP",
                                "ATK",
                                "DEF",
                                "SP_ATK",
                                "SP_DEF",
                                "NORM_SCORE",
                            ]
                        )
                    ],
                    value="NORM_SCORE",
                    style={"width": "100%"},
                )
            ]
        ),
        dcc.Graph("dist-2-chart", style={"width": "100%", "display": "inline-block"}),
        html.Div(
            children=dcc.Graph(
                id="type-rank-graph",
                figure=fig1,
            ),
            style={"width": "50%", "display": "inline-block"},
        ),
        html.Div(
            children=dcc.Graph(
                id="pokemon-rank-graph",
                figure=fig2,
            ),
            style={"width": "50%", "display": "inline-block"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="poke-chart",
                    options=[
                        {"label": i, "value": i}
                        for i in sorted(data["TYPE1"].unique().tolist())
                    ],
                    value="Dragon",
                    style={"width": "100%"},
                )
            ]
        ),
        dcc.Graph(
            "poke-type-graph", config={"displayModeBar": False}, style={"width": "100%"}
        ),
    ],
    style={"backgroundColor": "#D6EAF8", "textAlign": "center"},
)


@app.callback(Output("poke-type-graph", "figure"), [Input("poke-chart", "value")])
def update_graph(Type):
    updated = pokemon_by_type_graph(Type)
    return updated


@app.callback(Output("dist-1-chart", "figure"), [Input("dist-1-chart-menu", "value")])
def update_graph(attribute):
    updated = make_hist(attribute)
    return updated


@app.callback(Output("dist-2-chart", "figure"), [Input("dist-2-chart-menu", "value")])
def update_graph(attribute):
    updated = make_hist(attribute)
    return updated


app.run_server()