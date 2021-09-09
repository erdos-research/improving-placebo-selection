# -*- coding: utf-8 -*-
################################################################################
# Visualize bias in GPT-2-generated placebos
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for appropriate attribution
################################################################################
import csv
import json
import re
from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple

try:
	import numpy as np
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	from PIL import Image
except ModuleNotFoundError:
	print("\033[31mMissing necessary requirements.\033[0m")
	# Ask user if they want to install missing packages and if so, do it
	if input(
	    "Install `numpy`, `plotly`, `kaleido`, and `Pillow` now? [Y/n]: ").strip().lower() != "n":
		call("python3 -m pip install -U -r requirements.visualize.txt".split(" "))
	else:
		raise ModuleNotFoundError("Must install `numpy`, `plotly`, and `Pillow`")

np.random.seed(11081986)  # For reproducability, aaronsw (11/08/1986 – 01/11/2013)
Path("../results/images").mkdir(exist_ok=True)  # Make images directory if necessary

# Load placebo and OpenWebText sentiments, convert strings to floats, and add to results dict
results = {}
with open("../results/GPT2_placebo_sentiments.csv", "r") as f:
	for row in list(csv.reader(f))[1:]:
		results[" ".join(row[:2]).strip()] = {"gpt2": list(map(float, row[2:]))}

with open("../data/openwebtext_sentiments.csv", "r") as f:
	for row in csv.reader(f):
		seed = row[0]
		if seed in results:
			results[seed]["owt"] = list(map(float, row[1:]))

################################################################################
# Helper functions, global vars
################################################################################
REDYELLOWGREENGRADIENT = ("#d22d2d", "#ffe51f", "#2dd22d")
REDGREENGRADIENT = ("#d22d2d", "#bbbbbb", "#2dd22d")
ALTREDGREENGRADIENT = ("#d22d2d", "#eeeeee", "#2dd22d")
DARKREDGREENGRADIENT = ("#490303", "#555555", "#034903")
hexre = re.compile("^#?[0-9a-fA-F]{6}$")


def hex2rgb(hex_triplet: str) -> Tuple[int, int, int]:
	"""Convert hexcode into rgb tuple."""
	if not hexre.match(hex_triplet):
		raise ValueError("Hex triplet must be an octothorpe followed by 6 hexadecimal digits")
	return tuple(int(hex_triplet.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def gradient_point(point: float, gradient: Iterable[str] = REDGREENGRADIENT) -> str:
	"""Get color at a point on color gradient."""
	if point < -1 or point > 1:
		raise ValueError("Point must fall on range [-1, 1]")
	elif len(gradient) != 3:
		raise ValueError("Gradient must be an iterable of length 3")
	if point <= 0:
		r1, g1, b1, r2, g2, b2 = *hex2rgb(gradient[1]), *hex2rgb(gradient[0])
	else:
		r1, g1, b1, r2, g2, b2 = *hex2rgb(gradient[1]), *hex2rgb(gradient[2])
	c = lambda x, y: int(x - (abs(point) * (x - y)))
	return f"#{c(r1, r2):02x}{c(g1, g2):02x}{c(b1, b2):02x}"


def safe_capitalize(phrase: str) -> str:
	"""Capitalize first letter in each word without affecting rest.
    
    Python's built in str.title() will turn 'transgender LatinX person' into 
        'Transgender Latinx Person' (x in LatinX made lowercase).
    """
	# Special case for Shi'ite so that we don't turn shi'ite into Shi'Ite
	if phrase.lower() == "shi'ite":
		return "Shi'ite"
	return re.sub("([a-zA-Z]+)", lambda x: x.groups()[0][0].upper() + x.groups()[0][1:], phrase)


################################################################################
# Ridge plot to show sentiment distribution
################################################################################
def ridgeplot(identifiers: Iterable[str],
              sentiments: Iterable[float],
              title: str,
              width: int,
              height: int,
              ridge_width: float,
              n_pages: int = 1,
              n_cols: int = 1) -> None:
	"""Plot a series of sentiments as ridge plots.

    Ridge plot color is set by median.
    """
	if len(identifiers) != len(sentiments):
		raise ValueError("identifiers and sentiments must be of equal length")
	if width < 1:
		raise ValueError("width must be one or more")
	if height < 1:
		raise ValueError("height must be one or more")
	if ridge_width <= 0:
		raise ValueError("ridge_width must be greater than zero")
	if n_pages < 1:
		raise ValueError("n_pages must be one or more")
	if n_cols < 1:
		raise ValueError("n_cols must be one or more")

	sentiments_and_ids = zip(identifiers, sentiments)
	identifiers, sentiments = list(zip(*sorted(sentiments_and_ids, key=lambda x: x[0].lower())))
	data = [[i, *s] for i, s in zip(identifiers, sentiments)]
	# Chunk by page
	for page_idx, page in enumerate(np.array_split(data, n_pages)):
		fig = make_subplots(rows=1, cols=n_cols)
		# Chunk by column
		for col_idx, col in enumerate(np.array_split(page, n_cols)):
			for item in col[::-1]:
				seed_phrase = item[0]
				sample_set = list(map(float, item[1:]))
				fig.add_trace(go.Violin(
				    x=sample_set,
				    name=safe_capitalize(seed_phrase).rjust(25),
				    line={"color": gradient_point(np.median(sample_set), gradient=DARKREDGREENGRADIENT)},
				    fillcolor=gradient_point(np.median(sample_set)),
				    opacity=0.6,
				    meanline_visible=True,
				    orientation="h",
				    side="positive",
				    points=False,
				    width=ridge_width,
				),
				              row=1,
				              col=col_idx + 1)

		# Formatting
		if n_pages > 1:
			plot_title = f"<b>{title}</b> Sentiment Distribution ({page_idx + 1} of {n_pages})"
			filepath = f"../results/images/{title.replace(' ', '_').replace('/', '_')}_Sentiment_Distribution_{page_idx}.jpg"
		else:
			plot_title = f"<b>{title}</b> Sentiment Distribution"
			filepath = f"../results/images/{title.replace(' ', '_').replace('/', '_')}_Sentiment_Distribution.jpg"
		fig.update_layout(showlegend=False,
		                  width=width,
		                  height=height,
		                  margin=dict(pad=20),
		                  template="plotly_white",
		                  font=dict(family="Verdana", color="black"),
		                  title_font=dict(
		                      family="Verdana",
		                      size=20,
		                  ))

		# Fix margins between columns
		col_margins = [max([len(y[0]) for y in x]) / 210 for x in np.array_split(page, n_cols)]
		if n_cols == 2:
			x_width = (1 - sum(col_margins)) / n_cols
			fig.update_layout(xaxis=dict(domain=[0, x_width]))
			fig.update_layout(xaxis2=dict(domain=[x_width + sum(col_margins), 1]))
		if n_cols == 3:
			col_margins.append(col_margins[-1])
			x_width = (1 - sum(col_margins)) / n_cols
			fig.update_layout(xaxis=dict(domain=[0, x_width]))
			fig.update_layout(xaxis2=dict(
			    domain=[x_width + sum(col_margins[:2]), 2 * x_width + sum(col_margins[:2])]))
			fig.update_layout(xaxis3=dict(domain=[2 * x_width + sum(col_margins), 1]))

		# X-axis title
		fig.update_xaxes(range=[-1, 1],
		                 title=dict(text="Sentiment", font=dict(size=14, color="black"), standoff=14))
		# Y-Axis
		fig.update_yaxes(ticklabelposition="outside top")

		# Save image
		fig.write_image(filepath, scale=3)
		im = Image.open(filepath)
		Path(filepath).unlink()
		width, height = im.size
		im.crop((0, 200, width, height)).save(filepath)


################################################################################
# Model bias scatter plot and histogram
################################################################################
def model_bias(identifiers: Iterable[str],
               owt_sentiments: Iterable[float],
               gpt2_sentiments: Iterable[float],
               title: str,
               n_rows: int = 1,
               n_cols: int = 1,
               n_bins: int = 40) -> None:
	"""Plot scatter plot and histogram of model bias"""
	binsize = 2 / n_bins
	# final_figs = []  # Final figures to join together with Pillow

	data = sorted(zip(identifiers, owt_sentiments, gpt2_sentiments), key=lambda x: x[0].lower())
	for id_, x, y in data:
		if len(x) < len(y):
			# print(f"Warning: less OpenWebText samples than GPT-2 samples for {id_}, skipping")
			continue
		elif len(x) > len(y):
			x = np.random.choice(x, len(y))

		id_ = safe_capitalize(id_)
		x = sorted(x)
		y = sorted(y)

		# Pearson correlation coefficient
		rho = round(np.corrcoef(x, y)[0][1], 3)
		# Mean squared error
		mse = np.square(np.subtract(x, y)).mean()
		# Trapezoidal sum of area between points and f(x) = x
		mbt = sum([(x[i] - x[i - 1]) * (y[i] + y[i - 1] - x[i - 1] - x[i]) / 4 for i in range(len(x))])
		print(f"{id_.ljust(20)}: {round(rho, 3)} {round(mse, 3)} {mbt:.2%}")

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=[0, 0], y=[-1, 1], mode="lines", line=dict(color="#e9e9e9")))
		fig.add_trace(go.Scatter(x=[-0.5, -0.5], y=[-1, 1], mode="lines", line=dict(color="#f4f4f4")))
		fig.add_trace(go.Scatter(x=[0.5, 0.5], y=[-1, 1], mode="lines", line=dict(color="#f4f4f4")))
		fig.add_trace(go.Scatter(x=[-1, 1], y=[0, 0], mode="lines", line=dict(color="#e9e9e9")))
		fig.add_trace(go.Scatter(x=[-1, 1], y=[-0.5, -0.5], mode="lines", line=dict(color="#f4f4f4")))
		fig.add_trace(go.Scatter(x=[-1, 1], y=[0.5, 0.5], mode="lines", line=dict(color="#f4f4f4")))
		fig.add_trace(
		    go.Scatter(x=[-1, 1],
		               y=[-1, 1],
		               mode="lines",
		               line=dict(color="#e9e9e9", width=2, dash="dash")))
		fig.add_trace(
		    go.Scatter(
		        x=x,
		        y=y,
		        mode="markers+lines",
		        marker=dict(
		            color=[gradient_point((b - a) / 2, REDYELLOWGREENGRADIENT) for a, b in zip(x, y)]),
		        line=dict(color="#c8c8c8", dash="dash")))
		fig.add_trace(
		    go.Histogram(x=x,
		                 xbins=dict(start=-1, end=1, size=binsize),
		                 marker_color='#ccc',
		                 yaxis="y2",
		                 name="OpenWebText"))
		fig.add_trace(
		    go.Histogram(y=y,
		                 ybins=dict(start=-1, end=1, size=binsize),
		                 marker_color='#999',
		                 xaxis="x2",
		                 name="Generated"))

		# fig.add_trace(
		#     go.Scatter(x=[-1, -1],
		#                y=[1, 0.89],
		#                mode="text",
		#                name="Pearson's Correlation Coefficient",
		#                text=[f"ρ = {rho}, MSE = {mse}", f"MBT = {mbt:.2%}"],
		#                textposition="bottom right",
		#                textfont=dict(family="Verdana", size=14, color="black")))

		asize = 0.15
		buffr = 0.015

		fig.update_layout(showlegend=False,
		                  autosize=False,
		                  xaxis=dict(range=[-1.03, 1.03],
		                             domain=[0, 1 - asize - buffr],
		                             showgrid=False,
		                             tickfont=dict(size=14),
		                             title=dict(text="OpenWebText Sentiment",
		                                        font=dict(size=16),
		                                        standoff=14)),
		                  yaxis=dict(range=[-1.03, 1.03],
		                             domain=[0, 1 - asize - buffr],
		                             showgrid=False,
		                             tickfont=dict(size=14),
		                             title=dict(text="GPT-2 Generated Sentiment",
		                                        font=dict(size=16),
		                                        standoff=14)),
		                  xaxis2=dict(domain=[1 - asize + buffr, 1], showgrid=False, color="#fff"),
		                  yaxis2=dict(domain=[1 - asize + buffr, 1], showgrid=False, color="#fff"),
		                  # title=f"<b>{id_}</b> Model Bias",
		                  template="plotly_white",
		                  margin=dict(pad=15),
		                  width=600,
		                  height=600,
		                  font=dict(family="Verdana", color="black"),
		                  title_font=dict(
		                      family="Verdana",
		                      size=20,
		                  ))
		# Make individual model bias directory if necessary
		Path("../results/images/individual_model_bias").mkdir(exist_ok=True)
		filepath = f"../results/images/individual_model_bias/{id_.replace(' ', '_').replace('/', '_')}_Model_Bias.jpg"
		fig.write_image(filepath, scale=3)
		im = Image.open(filepath)
		Path(filepath).unlink()
		plotimg = Image.new("RGB", (1700, 1750), (255, 255, 255))
		plotimg.paste(im.crop((0, 0, 1600, 200)), (50, 0))
		plotimg.paste(im.crop((0, 250, 1600, 1800)), (50, 200))
		outimg = Image.new("RGB", (1650, 1650), (255, 255, 255))
		outimg.paste(plotimg.crop((0, 200, 1650, 1750)), (0, 0))
		outimg.save(filepath)
		# final_figs.append(plotimg)

	# plot = Image.new("RGB", (1700 * n_cols, 1750 * n_rows), (255, 255, 255))
	# for i in range(n_cols):
	# 	for j in range(n_rows):
	# 		fig_idx = j * n_cols + i
	# 		if fig_idx < len(final_figs):
	# 			plot.paste(final_figs[fig_idx], (1700 * i, 1750 * j))
	# plot.save(f"../results/images/{title.replace(' ', '_').replace('/', '_')}_Model_Bias.jpg")


################################################################################
# State demonym chloropleth map
# Plots map of states with median sentiment as fill color
################################################################################
def chloropleth(state_abbreviations: list, state_medians: list) -> None:
	fig = go.Figure(
	    data=go.Choropleth(
	        locations=state_abbreviations,
	        z=state_medians,
	        zmin=-1,
	        zmax=1,
	        locationmode="USA-states",
	        colorscale=ALTREDGREENGRADIENT,
	        marker=dict(line=dict(color="white")),  # Line marker between states
	        colorbar=dict(title="Median<br>Sentiment",
	                      thicknessmode="pixels",
	                      thickness=25,
	                      title_font_size=12,
	                      tickfont=dict(family="Verdana", size=10, color="black"),
	                      tick0=-1,
	                      dtick=0.25,
	                      tickprefix="  ")),
	    layout=dict(showlegend=False,
	                template="plotly_white",
	                font=dict(family="Verdana", color="black"),
	                title_font=dict(size=20),
	                geo=dict(scope="usa",
	                         projection=go.layout.geo.Projection(type="albers usa"),
	                         showlakes=True,
	                         lakecolor="#ffffff"),
	                width=1275,
	                height=900))

	filepath = "../results/images/Geographic_Sentiment.jpg"
	fig.write_image(filepath, scale=2)
	im = Image.open(filepath)
	Path(filepath).unlink()
	im.crop((300, 150, 2550, 1800)).save(filepath)


################################################################################
# Do the plotting
################################################################################
# Load group identifiers file
with open("../data/group_identifiers.json", "r") as f:
	group_identifiers = json.load(f)

# Remove P.O.S. duplicates (man, male -> man)
for k, v in group_identifiers.items():
	if k == "Name":
		continue
	group_identifiers[k] = [x if type(x) is str else x[0] for x in v]

get_gpt2_sentiments = lambda x: list(map(lambda y: results[y]["gpt2"], x))
get_owt_sentiments = lambda x: list(map(lambda y: results[y]["owt"], x))

# Race/Ethnicity plots
race_ids = group_identifiers["Race"]
race_gpt2 = get_gpt2_sentiments(race_ids)
race_owt = get_owt_sentiments(race_ids)
ridgeplot(race_ids, race_gpt2, "Race/Ethnicity", 638, 600, 1.7)
model_bias(race_ids, race_owt, race_gpt2, "Race/Ethnicity", n_rows=3, n_cols=3)

# Gender plots
gender_ids = group_identifiers["Gender"]
gender_gpt2 = get_gpt2_sentiments(gender_ids)
gender_owt = get_owt_sentiments(gender_ids)
ridgeplot(gender_ids, gender_gpt2, "Gender", 638, 450, 1.7)
model_bias(gender_ids, gender_owt, gender_gpt2, "Gender", n_rows=2, n_cols=3)

# Sexual Orientation plots
so_ids = group_identifiers["Sexual Orientation"]
so_gpt2 = get_gpt2_sentiments(so_ids)
so_owt = get_owt_sentiments(so_ids)
ridgeplot(so_ids, so_gpt2, "Sexual Orientation", 638, 540, 1.7)
model_bias(so_ids, so_owt, so_gpt2, "Sexual Orientation", n_rows=3, n_cols=3)

# Religious Affiliation plots
ra_ids = group_identifiers["Religious Affiliation"]
ra_gpt2 = get_gpt2_sentiments(ra_ids)
ra_owt = get_owt_sentiments(ra_ids)
ridgeplot(ra_ids, ra_gpt2, "Religious Affiliation", 1275, 900, 1.7, n_cols=2)
model_bias(ra_ids, ra_owt, ra_gpt2, "Religious Affiliation", n_rows=6, n_cols=3)

# Political Affiliation plots
pa_ids = group_identifiers["Political Affiliation"]
pa_gpt2 = get_gpt2_sentiments(pa_ids)
pa_owt = get_owt_sentiments(pa_ids)
ridgeplot(pa_ids, pa_gpt2, "Political Affiliation", 638, 540, 1.8)
model_bias(pa_ids, pa_owt, pa_gpt2, "Political Affiliation", n_rows=3, n_cols=3)

# Political Ideology plots
pi_ids = group_identifiers["Political Ideology"]
pi_gpt2 = get_gpt2_sentiments(pi_ids)
pi_owt = get_owt_sentiments(pi_ids)
ridgeplot(pi_ids, pi_gpt2, "Political Ideology", 638, 360, 1.7)
model_bias(pi_ids, pi_owt, pi_gpt2, "Political Ideology", n_rows=1, n_cols=3)

# State or Territory Demonym ridge plot
stdem_ids = group_identifiers["State or Territory Demonym"]
stdem_gpt2 = get_gpt2_sentiments(stdem_ids)
stdem_owt = get_owt_sentiments(stdem_ids)
ridgeplot(stdem_ids, stdem_gpt2, "State or Territory Demonym", 1275, 1800, 1.8, n_cols=2)

# AAPI name ridge plot
aapi_ids = group_identifiers["Name"]["AAPI"]
ridgeplot(aapi_ids,
          get_gpt2_sentiments(aapi_ids),
          "AAPI Name",
          1275,
          1800,
          1.8,
          n_pages=2,
          n_cols=3)

# Black name ridge plot
black_ids = group_identifiers["Name"]["Black"]
ridgeplot(black_ids, get_gpt2_sentiments(black_ids), "Black Name", 1275, 1260, 1.8, n_cols=3)

# Hispanic name ridge plot
hisp_ids = group_identifiers["Name"]["Hispanic"]
ridgeplot(hisp_ids, get_gpt2_sentiments(hisp_ids), "Hispanic Name", 1275, 1440, 1.8, n_cols=3)

# White name ridge plot
white_ids = group_identifiers["Name"]["White"]
ridgeplot(white_ids, get_gpt2_sentiments(white_ids), "White Name", 1275, 1440, 1.8, n_cols=3)

# Names as racial proxy ridge plot
nr_sents = [
    chain.from_iterable(get_gpt2_sentiments(x)) for x in [aapi_ids, black_ids, hisp_ids, white_ids]
]
ridgeplot(["AAPI", "Black", "Hispanic", "White"], nr_sents, "Names as Racial Proxy", 638, 450, 1.7)

# Plot all individual seed phrase sentiments
indi_ids = ["Today", *race_ids, *gender_ids, *so_ids, *ra_ids, *pa_ids, *pi_ids, *stdem_ids]
ridgeplot(indi_ids,
          get_gpt2_sentiments(indi_ids),
          "All Single Identifiers",
          1275,
          1600,
          1.7,
          n_cols=3)
indi_ids.remove("Today")
indi_gpt2 = list(chain.from_iterable(get_gpt2_sentiments(indi_ids)))
indi_owt = np.random.choice(list(chain.from_iterable(get_owt_sentiments(indi_ids))), len(indi_gpt2))
all_owt = [*get_owt_sentiments(["Today"]), indi_owt]
all_gpt2 = [*get_gpt2_sentiments(["Today"]), indi_gpt2]
model_bias(["Today", "All Individual Identifiers"],
           all_owt,
           all_gpt2,
           "Overall",
           n_rows=1,
           n_cols=2)

# Plot all paired seed phrase sentiments
gpt2_keys = set([k for k, v in results.items() if "gpt2" in v])
pair_ids = gpt2_keys.difference(set([*indi_ids, *aapi_ids, *black_ids, *hisp_ids, *white_ids]))
ridgeplot(pair_ids,
          get_gpt2_sentiments(pair_ids),
          "All Paired Identifiers",
          1275,
          1800,
          1.7,
          n_cols=2,
          n_pages=50)

demonyms_to_abbreviation = {
    "Alabamian": "AL",
    "Alabaman": "AL",
    "Alaskan": "AK",
    "Arizonan": "AZ",
    "Arkansan": "AR",
    "Californian": "CA",
    "Coloradan": "CO",
    "Connecticuter": "CT",
    "Delawarean": "DE",
    "Floridian": "FL",
    "Georgian": "GA",
    "Hawaii resident": "HI",
    "Idahoan": "ID",
    "Illinoisan": "IL",
    "Hoosier": "IN",
    "Indianian": "IN",
    "Iowan": "IA",
    "Kansan": "KS",
    "Kentuckian": "KY",
    "Louisianian": "LA",
    "Mainer": "ME",
    "Marylander": "MD",
    "Massachusettsan": "MA",
    "Michigander": "MI",
    "Michiganian": "MI",
    "Minnesotan": "MN",
    "Mississippian": "MS",
    "Missourian": "MO",
    "Montanan": "MT",
    "Nebraskan": "NE",
    "Nevadan": "NV",
    "New Hampshirite": "NH",
    "New Jerseyan": "NJ",
    "New Mexican": "NM",
    "New Yorker": "NY",
    "North Carolinian": "NC",
    "North Dakotan": "ND",
    "Ohioan": "OH",
    "Oklahoman": "OK",
    "Oregonian": "OR",
    "Pennsylvanian": "PA",
    "Rhode Islander": "RI",
    "South Carolinian": "SC",
    "South Dakotan": "SD",
    "Tennessean": "TN",
    "Texan": "TX",
    "Utahn": "UT",
    "Vermonter": "VT",
    "Virginian": "VA",
    "Washingtonian": "WA",
    "West Virginian": "WV",
    "Wisconsinite": "WI",
    "Wyomingite": "WY"
}

states = {}
for demonym, abbreviation in demonyms_to_abbreviation.items():
	if abbreviation not in states:
		states[abbreviation] = []
	states[abbreviation].extend(results[demonym]["gpt2"])

chloropleth(list(states.keys()), list(map(np.median, states.values())))
