In this repository, you will find our project for Webscrapping & Applied Machine Learning.
We had to propose a solution/platform answering to tourism and RSE problematics.
To do so with decided to build a trip recommandation application. 
To answer to the RSE problematic, we wanted to help reduce emissionsfrom tourist travels. To do so, we scrapped data from the SNCF API to get the CO2 emissions for most of the available trips on their platforms.
We then scrapped reviews from tripadvisor from all the cities that had reviews on this website. We encounteres issued on that aspect due to tripadvisor protecting its data, and we got detected as bots too easily, so our data is reduced.
We then trained several models on that data to use it in our application. 
We chose to build an application that takes as input a prompt, describing a past trip, or a desired trip in the form of a review. Our application then returns a list of the top 3 cities that are the most likely to be enjoyed by the user based on their prompt, and compared to the tripadvisord reviews. 
To integrate the train emissions, we then make a recommandation based on the balance between the train emissions of the city, and the similarity score between the prompt and the cities.
This is supposed to help us give the best recommandation from an environmental (and usually economic) point of view, and from a user satisfaction point of view. Tha application is in a streamlit format, contained in the last cell of our ML notebook.
