# EchosounderNetCDF2LSSS
This repository comprises of two main functions:
- 1. Reads and writes the interpreation masks for fisheries acoustic
  - The mask will be region based and the grid  will be time and depth. The sound speed used to calculate the depth will be stored as metadata. 
  - The region will be defined to include both _in_ and _on_ the region outline. Donut shapes will be supported.
  - Background information can be found here: [https://docs.google.com/document/d/1F5ub9-ElnGWgoFzOhwrNiAB6fZRhKI8Nw6FskMhzI0g/edit#heading=h.ihw2gdxqw9td]
- 2. Post the regions and masks to LSSS via API
