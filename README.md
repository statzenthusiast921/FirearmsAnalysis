# FirearmsAnalysis

### Description

The purpose of this project is to:
- 1.) Explore the similarity in the text of the firearm laws and determine if any regional relationships exist.
- 2.) Discover patterns in the text and determine the relative importance of certain words.
- 3.) Determine if any latent groupings of laws exist when including regional, demographic data as well as sentiment data.

### Link

- [Firearms Dashboard](https://firearm-dashboard-app.fly.dev/)
  - Using Free Tier on fly.io --> very slow, please give ~5 min to load all charts

### Data

The data used for this analysis included:
- [RAND State Firearm Law Database](https://www.rand.org/pubs/tools/TLA243-2-v2.html)
- [RAND State-Level Estimates of Household Firearm Ownership](https://www.rand.org/pubs/tools/TL354.html)
- scraped data from [State Firearm Death Statistics](https://www.statefirearmlaws.org/states/)
- scraped data from [Federal Reserve Bank of St. Louis Economic Data (1)](https://fred.stlouisfed.org/release/tables?rid=118&eid=259194)
- scraped data from [Federal Reserve Bank of St. Louis Economic Data (2)](https://fred.stlouisfed.org/release/tables?eid=259515&rid=249)
(https://github.com/statzenthusiast921/US_Elections_Project/blob/main/Data/FullElectionsData.xlsx)



### Challenges
- Finding demographic data that covered the same years as the firearm legislation proved to be a difficult task.  I included data spanning as many years as I was able to find.
- Heroku was my preferred choice for hosting personal projects, but as November 28th, 2022, Heroku's free services will be discontuined.  Finding an easy-to-use, free, and comparable service was challenging.  I eventually settled on [fly.io](fly.io) as it was relatively easy to set up using their guides.
