# -*- coding: utf-8 -*-
################################################################################
# Visualize bias in GPT-2 and GPT-3.5 Turbo-generated placebos
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

# Get current working directory
cwd = Path(__file__).parent.resolve()

try:
	import numpy as np
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	from PIL import Image
except ModuleNotFoundError:
	print("\033[31mMissing necessary requirements.\033[0m")
	# Ask user if they want to install missing packages and if so, do it
	visualize_requirements_fp = (cwd / "requirements.generate.txt").resolve()
	if input("Install `numpy`, `plotly`, and `Pillow` now? [Y/n]: ").strip().lower() != "n":
		call(f"python3 -m pip install -U -r {visualize_requirements_fp}".split(" "))
	else:
		raise ModuleNotFoundError("Must install `numpy`, `plotly`, and `Pillow`")

Image.MAX_IMAGE_PIXELS = None  # Allow large images to be written
np.random.seed(11081986)  # For reproducability, aaronsw (11/08/1986 – 01/11/2013)
(cwd / "../results/images").resolve().mkdir(exist_ok=True)  # Make images directory if necessary

# Load placebo and OpenWebText sentiments, convert strings to floats, and add to results dict
results = {}
with open((cwd / "../results/GPT2_placebo_sentiments.csv").resolve(), "r") as f:
	for row in list(csv.reader(f))[1:]:
		results[" ".join(row[:2]).strip()] = {"gpt2": list(map(float, row[2:]))}

with open((cwd / "../results/GPT3_5_placebo_sentiments.csv").resolve(), "r") as f:
	for row in list(csv.reader(f))[1:]:
		results[" ".join(row[:2]).strip()]["gpt3.5-turbo"] = list(map(float, row[2:]))

with open((cwd / "../data/openwebtext_sentiments.csv").resolve(), "r") as f:
	for row in csv.reader(f):
		seed = row[0]
		if seed in results:
			results[seed]["owt"] = list(map(float, row[1:]))

import json
with open("results.json", "w") as f:
	json.dump(results, f, indent=4)

################################################################################
# Helper functions, global vars
################################################################################
REDYELLOWGREENGRADIENT = ("#d22d2d", "#ffe51f", "#2dd22d")
BLUEPINKGRADIENT = ("#d16b6b", "#e6d660", "#5eb85e")
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
def split_list(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def ridgeplot(identifiers: Iterable[str],
              gpt2_sentiments: Iterable[float],
              gpt3_5_sentiments: Iterable[float],
              title: str,
              width: int,
              height: int,
              ridge_width: float,
              n_pages: int = 1,
              n_cols: int = 1) -> None:
	"""Plot a series of sentiments as ridge plots.

    Ridge plot color is set by median.
    """
	if len(identifiers) != len(gpt2_sentiments):
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

	# Ensure that the data is of the same length and sorted by identifier
	sentiments_and_ids = zip(identifiers, gpt2_sentiments, gpt3_5_sentiments)
	identifiers, gpt2_sentiments, gpt3_5_sentiments = list(
	    zip(*sorted(sentiments_and_ids, key=lambda x: x[0].lower())))
	data = [[i, s2, s3_5] for i, s2, s3_5 in zip(identifiers, gpt2_sentiments, gpt3_5_sentiments)]
	# Chunk by page
	for page_idx, page in enumerate(split_list(data, n_pages)):
		fig = make_subplots(rows=1, cols=n_cols)
		# Chunk by column
		for col_idx, col in enumerate(split_list(page, n_cols)):
			for item in col[::-1]:
				seed_phrase = item[0]
				sample_set_gpt2 = list(map(float, item[1]))  # gpt2 sentiments
				sample_set_gpt3_5 = list(map(float, item[2]))  # gpt3_5 sentiments

				# add GPT 2 sentiments as a trace
				fig.add_trace(go.Violin(
				    x=sample_set_gpt2,
				    name=safe_capitalize(seed_phrase).rjust(25),
				    line={
				        "color": gradient_point(np.median(sample_set_gpt2), gradient=DARKREDGREENGRADIENT),
				        "width": 6,
				    },
				    fillcolor=gradient_point(np.median(sample_set_gpt2)),
				    opacity=0.6,
				    meanline_visible=True,
				    orientation="h",
				    side="positive",
				    points=False,
				    width=ridge_width,
				),
				              row=1,
				              col=col_idx + 1)

				# add GPT 3.5-Turbo sentiments as a trace
				fig.add_trace(go.Violin(
				    x=sample_set_gpt3_5,
				    name=safe_capitalize(seed_phrase).rjust(25),
				    line={
				        "color": "#81badb",
				        "width": 6
				    },
				    fillcolor="#9ceafc",
				    opacity=0.4,
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
			filepath = (
			    cwd /
			    f"../results/images/{title.replace(' ', '_').replace('/', '_')}_Sentiment_Distribution_{page_idx}.jpg"
			).resolve()
		else:
			plot_title = f"<b>{title}</b> Sentiment Distribution"
			filepath = (
			    cwd /
			    f"../results/images/{title.replace(' ', '_').replace('/', '_')}_Sentiment_Distribution.jpg"
			).resolve()
		fig.update_layout(showlegend=False,
		                  width=width,
		                  height=height,
		                  margin=dict(pad=30),
		                  template="plotly_white",
		                  font=dict(family="Verdana", color="black", size=36),
		                  title_font=dict(
		                      family="Verdana",
		                      size=60,
		                  ))

		# Fix margins between columns
		col_margins = [max([len(y[0]) for y in x]) / 210 for x in split_list(page, n_cols)]
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
		                 title=dict(text="Sentiment", font=dict(size=42, color="black"), standoff=42))
		# Y-Axis
		fig.update_yaxes(ticklabelposition="outside top")

		# Write formatted image
		fig.write_image(filepath)
		im = Image.open(filepath)
		Path(filepath).unlink()
		width, height = im.size
		im = im.crop((0, 50, width, height))
		# Composite legend from file
		legend = Image.open(cwd / "ridgeplot_legend.jpg")
		im2 = Image.new("RGB", (im.size[0] + legend.size[0], im.size[1]), (255, 255, 255))
		im2.paste(im, (0, 0))
		im2.paste(legend, (im.size[0], 0))
		im2.save(filepath)


################################################################################
# Model bias scatter plot and histogram
################################################################################
def model_bias(identifiers: Iterable[str],
               owt_sentiments: Iterable[float],
               gpt2_sentiments: Iterable[float],
               gpt3_5_sentiments: Iterable[float],
               title: str,
               n_rows: int = 1,
               n_cols: int = 1,
               n_bins: int = 40) -> None:
	"""Plot scatter plot and histogram of model bias"""
	binsize = 2 / n_bins
	# final_figs = []  # Final figures to join together with Pillow

	data = sorted(zip(identifiers, owt_sentiments, gpt2_sentiments, gpt3_5_sentiments),
	              key=lambda x: x[0].lower())
	for id_, x, y, z in data:
		id_ = safe_capitalize(id_)
		min_length = min(len(x), len(y), len(z))
		x = sorted(np.random.choice(x, min_length))
		y = sorted(np.random.choice(y, min_length))
		z = sorted(np.random.choice(z, min_length))

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
		    go.Scatter(x=x,
		               y=z,
		               mode="markers+lines",
		               marker=dict(
		                   color=[gradient_point((b - a) / 2, BLUEPINKGRADIENT) for a, b in zip(x, y)],
		                   symbol="triangle-up",
		                   opacity=0.5),
		               line=dict(color="#e8e8e8", width=1)))
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
		                 name="GPT-2 Generated"))

		asize = 0.15
		buffr = 0.015

		fig.update_layout(
		    showlegend=False,
		    autosize=False,
		    xaxis=dict(range=[-1.03, 1.03],
		               domain=[0, 1 - asize - buffr],
		               showgrid=False,
		               tickfont=dict(size=14),
		               title=dict(text="OpenWebText Sentiment", font=dict(size=16), standoff=14)),
		    yaxis=dict(
		        range=[-1.03, 1.03],
		        domain=[0, 1 - asize - buffr],
		        showgrid=False,
		        tickfont=dict(size=14),
		        title=dict(
		            text="● = GPT-2 Generated Sentiment<br>▲ = GPT-3.5 Turbo Generated Sentiment",
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
		model_bias_directory = (cwd / "../results/images/individual_model_bias").resolve()
		model_bias_directory.mkdir(exist_ok=True)
		filepath = model_bias_directory / f"{id_.replace(' ', '_').replace('/', '_')}_Model_Bias.jpg"
		fig.write_image(filepath, scale=3)
		im = Image.open(filepath)
		Path(filepath).unlink()
		plotimg = Image.new("RGB", (1700, 1750), (255, 255, 255))
		plotimg.paste(im.crop((0, 0, 1600, 200)), (50, 0))
		plotimg.paste(im.crop((0, 250, 1600, 1800)), (50, 200))
		outimg = Image.new("RGB", (1650, 1650), (255, 255, 255))
		outimg.paste(plotimg.crop((0, 200, 1650, 1750)), (0, 0))
		outimg.save(filepath)


################################################################################
# State demonym chloropleth map
# Plots map of states with median sentiment as fill color
################################################################################
def chloropleth(state_abbreviations: list, state_medians: list, model: str) -> None:
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

	filepath = (cwd / f"../results/images/Geographic_Sentiment_{model}.jpg").resolve()
	fig.write_image(filepath, scale=2)
	im = Image.open(filepath)
	Path(filepath).unlink()
	im.crop((300, 150, 2550, 1800)).save(filepath)


################################################################################
# Do the plotting
################################################################################
# Load group identifiers file
with open((cwd / "../data/group_identifiers.json").resolve(), "r") as f:
	group_identifiers = json.load(f)

# Remove P.O.S. duplicates (man, male -> man)
for k, v in group_identifiers.items():
	if k == "Name":
		continue
	group_identifiers[k] = [x if type(x) is str else x[0] for x in v]

get_gpt2_sentiments = lambda x: list(map(lambda y: results[y]["gpt2"], x))
get_gpt3_5_sentiments = lambda x: list(map(lambda y: results[y]["gpt3.5-turbo"], x))
get_owt_sentiments = lambda x: list(map(lambda y: results[y]["owt"], x))

# Race/Ethnicity plots
race_ids = group_identifiers["Race"]
race_gpt2 = get_gpt2_sentiments(race_ids)
race_gpt3_5 = get_gpt3_5_sentiments(race_ids)
race_owt = get_owt_sentiments(race_ids)
ridgeplot(race_ids, race_gpt2, race_gpt3_5, "Race/Ethnicity", 638 * 3, 600 * 3, 1.7)
model_bias(race_ids, race_owt, race_gpt2, race_gpt3_5, "Race/Ethnicity", n_rows=3, n_cols=3)

# Gender plots
gender_ids = group_identifiers["Gender"]
gender_gpt2 = get_gpt2_sentiments(gender_ids)
gender_gpt3_5 = get_gpt3_5_sentiments(gender_ids)
gender_owt = get_owt_sentiments(gender_ids)
ridgeplot(gender_ids, gender_gpt2, gender_gpt3_5, "Gender", 638 * 3, 450 * 3, 1.7)
model_bias(gender_ids, gender_owt, gender_gpt2, gender_gpt3_5, "Gender", n_rows=2, n_cols=3)

# Sexual Orientation plots
so_ids = group_identifiers["Sexual Orientation"]
so_gpt2 = get_gpt2_sentiments(so_ids)
so_gpt3_5 = get_gpt3_5_sentiments(so_ids)
so_owt = get_owt_sentiments(so_ids)
ridgeplot(so_ids, so_gpt2, so_gpt3_5, "Sexual Orientation", 638 * 3, 540 * 3, 1.7)
model_bias(so_ids, so_owt, so_gpt2, so_gpt3_5, "Sexual Orientation", n_rows=3, n_cols=3)

# Religious Affiliation plots
ra_ids = group_identifiers["Religious Affiliation"]
ra_gpt2 = get_gpt2_sentiments(ra_ids)
ra_gpt3_5 = get_gpt3_5_sentiments(ra_ids)
ra_owt = get_owt_sentiments(ra_ids)
ridgeplot(ra_ids, ra_gpt2, ra_gpt3_5, "Religious Affiliation", 1275 * 3, 900 * 3, 1.7, n_cols=2)
model_bias(ra_ids, ra_owt, ra_gpt2, ra_gpt3_5, "Religious Affiliation", n_rows=6, n_cols=3)

# Political Affiliation plots
pa_ids = group_identifiers["Political Affiliation"]
pa_gpt2 = get_gpt2_sentiments(pa_ids)
pa_gpt3_5 = get_gpt3_5_sentiments(pa_ids)
pa_owt = get_owt_sentiments(pa_ids)
ridgeplot(pa_ids, pa_gpt2, pa_gpt3_5, "Political Affiliation", 638 * 3, 540 * 3, 1.8)
model_bias(pa_ids, pa_owt, pa_gpt2, pa_gpt3_5, "Political Affiliation", n_rows=3, n_cols=3)

# Political Ideology plots
pi_ids = group_identifiers["Political Ideology"]
pi_gpt2 = get_gpt2_sentiments(pi_ids)
pi_gpt3_5 = get_gpt3_5_sentiments(pi_ids)
pi_owt = get_owt_sentiments(pi_ids)
ridgeplot(pi_ids, pi_gpt2, pi_gpt3_5, "Political Ideology", 638 * 3, 360 * 3, 1.7)
model_bias(pi_ids, pi_owt, pi_gpt2, pi_gpt3_5, "Political Ideology", n_rows=1, n_cols=3)

# State or Territory Demonym ridge plot
stdem_ids = group_identifiers["State or Territory Demonym"]
stdem_gpt2 = get_gpt2_sentiments(stdem_ids)
stdem_gpt3_5 = get_gpt3_5_sentiments(stdem_ids)
ridgeplot(stdem_ids,
          stdem_gpt2,
          stdem_gpt3_5,
          "State or Territory Demonym",
          3825,
          5400,
          1.8,
          n_cols=2)

# AAPI name ridge plot
aapi_ids = group_identifiers["Name"]["AAPI"]
ridgeplot(aapi_ids,
          get_gpt2_sentiments(aapi_ids),
          get_gpt3_5_sentiments(aapi_ids),
          "AAPI Name",
          3825,
          5400,
          1.8,
          n_pages=2,
          n_cols=3)

# Black name ridge plot
black_ids = group_identifiers["Name"]["Black"]
ridgeplot(black_ids,
          get_gpt2_sentiments(black_ids),
          get_gpt3_5_sentiments(black_ids),
          "Black Name",
          3825,
          3780,
          1.8,
          n_cols=3)

# Hispanic name ridge plot
hisp_ids = group_identifiers["Name"]["Hispanic"]
ridgeplot(hisp_ids,
          get_gpt2_sentiments(hisp_ids),
          get_gpt3_5_sentiments(hisp_ids),
          "Hispanic Name",
          3825,
          4320,
          1.8,
          n_cols=3)

# White name ridge plot
white_ids = group_identifiers["Name"]["White"]
ridgeplot(white_ids,
          get_gpt2_sentiments(white_ids),
          get_gpt3_5_sentiments(white_ids),
          "White Name",
          3825,
          4320,
          1.8,
          n_cols=3)

# Names as racial proxy ridge plot
nr_gpt2_sents = [
    list(chain.from_iterable(get_gpt2_sentiments(x)))
    for x in [aapi_ids, black_ids, hisp_ids, white_ids]
]
shortest_nr_gpt2_sent = min([len(x) for x in nr_gpt2_sents])
nr_gpt2_sents = [np.random.choice(x, shortest_nr_gpt2_sent) for x in nr_gpt2_sents]
nr_gpt3_5_sents = [
    list(chain.from_iterable(get_gpt3_5_sentiments(x)))
    for x in [aapi_ids, black_ids, hisp_ids, white_ids]
]
shortest_nr_gpt3_5_sent = min([len(x) for x in nr_gpt3_5_sents])
nr_gpt3_5_sents = [np.random.choice(x, shortest_nr_gpt3_5_sent) for x in nr_gpt3_5_sents]
ridgeplot(["AAPI", "Black", "Hispanic", "White"], nr_gpt2_sents, nr_gpt3_5_sents,
          "Names as Racial Proxy", 1914, 1350, 1.7)

# Plot all individual seed phrase sentiments
indi_ids = ["Today", *race_ids, *gender_ids, *so_ids, *ra_ids, *pa_ids, *pi_ids, *stdem_ids]
ridgeplot(indi_ids,
          get_gpt2_sentiments(indi_ids),
          get_gpt3_5_sentiments(indi_ids),
          "All Single Identifiers",
          3825,
          4800,
          1.7,
          n_cols=3)
indi_ids.remove("Today")
indi_gpt2 = list(chain.from_iterable(get_gpt2_sentiments(indi_ids)))
indi_owt = np.random.choice(list(chain.from_iterable(get_owt_sentiments(indi_ids))), len(indi_gpt2))
all_owt = [*get_owt_sentiments(["Today"]), indi_owt]
all_gpt2 = [*get_gpt2_sentiments(["Today"]), indi_gpt2]
all_gpt3_5 = [*get_gpt3_5_sentiments(["Today"]), indi_gpt2]
model_bias(["Today", "All Individual Identifiers"],
           all_owt,
           all_gpt2,
           all_gpt3_5,
           "Overall",
           n_rows=1,
           n_cols=2)

# Plot all paired seed phrase sentiments
gpt2_keys = set([k for k, v in results.items() if "gpt2" in v])
pair_ids = gpt2_keys.difference(set([*indi_ids, *aapi_ids, *black_ids, *hisp_ids, *white_ids]))
pair_gpt2_sents = get_gpt2_sentiments(pair_ids)
pair_gpt3_5_sents = get_gpt3_5_sentiments(pair_ids)
shortest_pair_gpt2_sent = min([len(x) for x in pair_gpt2_sents])
shortest_pair_gpt3_5_sent = min([len(x) for x in pair_gpt3_5_sents])
pair_gpt2_sents = [np.random.choice(x, shortest_pair_gpt2_sent) for x in pair_gpt2_sents]
pair_gpt3_5_sents = [np.random.choice(x, shortest_pair_gpt3_5_sent) for x in pair_gpt3_5_sents]
ridgeplot(pair_ids,
          pair_gpt2_sents,
          pair_gpt3_5_sents,
          "All Paired Identifiers",
          3825,
          4800,
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

states_gpt2 = {}
for demonym, abbreviation in demonyms_to_abbreviation.items():
	if abbreviation not in states_gpt2:
		states_gpt2[abbreviation] = []
	states_gpt2[abbreviation].extend(results[demonym]["gpt2"])

chloropleth(list(states_gpt2.keys()), list(map(np.median, states_gpt2.values())), "GPT-2")

states_gpt3_5 = {}
for demonym, abbreviation in demonyms_to_abbreviation.items():
	if abbreviation not in states_gpt3_5:
		states_gpt3_5[abbreviation] = []
	states_gpt3_5[abbreviation].extend(results[demonym]["gpt3.5-turbo"])

chloropleth(list(states_gpt3_5.keys()), list(map(np.median, states_gpt3_5.values())),
            "GPT-3.5 Turbo")
