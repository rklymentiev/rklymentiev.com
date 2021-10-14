+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 10  # Order that this section will appear

title = ""
subtitle = ""

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "1"

[design.background]

  # Background image.
  image = "headers/altai.jpg"  # Name of image in `static/img/`.
  image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.
  image_size = "cover"  #  Options are `cover` (default), `contain`, or `actual` size.
  image_position = "center"  # Options include `left`, `center` (default), or `right`.
  image_parallax = true  # Use a fun parallax-like fixed background effect? true/false

  # Text color (true=light or false=dark).
  text_color_light = true

[design.spacing]
  # Customize the section spacing. Order is top, right, bottom, left.
  padding = ["20px", "0", "20px", "0"]

[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++

<style>
.box {
  width: absolute;
  margin: 50px auto;
  border: 4px solid #ffffff;
  padding: 20px;
  text-align: left;
  font-weight: 900;
  color: #ffffff;
  border-radius: 25px;
  position: relative;
}


.sb:after {
  content: '';
  position: absolute;
  display: block;
  width: 0;
  z-index: 1;
  border-style: solid;
  border-width: 0 0 20px 20px;
  border-color: transparent transparent #ffffff transparent;
  top: 79%;
  left: -20px;
  margin-top: -10px;
}
</style>


<div class="box sb">Hi there!

My name is Ruslan Klymentiev. Welcome to my blog. 

This is the place where I post stuff related to:

* Machine Learning
* Probability & Statistics
* Data Visualization
* Neuroscience
</div>


