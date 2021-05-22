#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
from geopy.geocoders import Nominatim 
import requests
from pandas.io.json import json_normalize 
import matplotlib.cm as cm
import matplotlib.colors as color
from sklearn.cluster import KMeans
import folium

from bs4 import BeautifulSoup
import requests

from IPython.display import display_html


# ## Webscraping
# 
# Webscraping for Data collection

# In[2]:


html='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
data  = requests.get(html).text
soup= BeautifulSoup(data,'html5lib')

tables= soup.find_all('table')
print(len(tables))


# In[3]:


Toronto= pd.read_html(str(tables[0]), flavor='bs4')


# ## Data Pre-processing 
# 

# In[4]:


column_names = ['Postal Code','Borough', 'Neighborhood']
df=pd.DataFrame(columns=column_names)


# In[5]:


r=[]
for i in range(0,9):
    a=Toronto[0][i]
    for j in range(len(a)):
        h=a[j]
        r.append(h)


# In[6]:


c=[]
d=[]
for i in range(len(r)):
    a=r[i]
    c.append(a[:3])
    d.append(a[3:])


# In[7]:


e=[]
x=[]
p=[]

for i in range(len(d)):
    s=d[i]
    s=s.split('(')
    e.append(s)
    if e[i][0]=='Not assigned':
        e[i].append('Not assigned')
    e[i].append(c[i])
    s=e[i][1]
    s=s.replace(')','')
    s=s.replace('/',',')
    e[i][1]=s


# In[8]:


for i in range(len(e)):
    borough=e[i][0]
    neighborhood_name=e[i][1]
    pos=e[i][2]
    df=df.append({'Borough': borough,'Neighborhood': neighborhood_name,'Postal Code':pos}, ignore_index=True)


# In[9]:


df=df.drop(index=df[df['Borough']=='Not assigned'].index)
df.reset_index().drop(columns='index')


# Finding Latitude and Longitude for addresses 
# 
# 

# In[35]:


import geocoder


# In[10]:


for i in df['Postal Code']:
    lat_lng_coords = None
    while(lat_lng_coords is None):
        g= geocoder.google(i+', Toronto, Ontario')
        lat_lng_coords = g.latlng
    df['Latitude'] = lat_lng_coords[0]
    df['Longitude'] = lat_lng_coords[1]


# **Note :** Since the geocode isn't returning the request I downloaded the csv file for Latitude & Longitude

# In[11]:


lalo=pd.read_csv('C:\\Users\\preda\\Downloads\\Geospatial_Coordinates.csv')
df=df.merge(lalo,on='Postal Code')
df


# ## Createing a Map of Toronto Neighbour

# In[12]:


add='Toronto'
geolocator=Nominatim(user_agent='toronto_explorer')
location=geolocator.geocode(add)
lon=location.longitude
lat=location.latitude


# In[13]:


Map=folium.Map(location=[lat,lon],zoom_start=12)

for la, lo, label in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [la, lo],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(Map)
Map


# ## Analysing Neighborhood  & Venues

# In[15]:


df1 = df[df['Borough'].str.contains('Toronto',regex=False)]
df1


# In[16]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[17]:


venues = getNearbyVenues(names=df['Neighborhood'],
                                   latitudes=df['Latitude'],
                                   longitudes=df['Longitude']
                                  )


# In[18]:


venues.groupby('Neighborhood').count()


# In[19]:


onehot = pd.get_dummies(venues[['Venue Category']], prefix="", prefix_sep="")
onehot['Neighborhood'] = venues['Neighborhood'] 
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]


# In[20]:


onehot.head()


# In[21]:


grouped = onehot.groupby('Neighborhood').mean().reset_index()


# In[22]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[23]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = grouped['Neighborhood']

for ind in np.arange(grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Clustering using KMeans 

# In[24]:


k = 10
clustering = grouped.drop('Neighborhood', 1)
kmeans = KMeans(n_clusters=k, random_state=5).fit(clustering)
kmeans.labels_[0:10] 


# In[25]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
#neighborhoods_venues_sorted[ 'Cluster Labels']=kmeans.labels_


# In[26]:


final= df1
final= df1.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
final


# In[27]:


Map2 = folium.Map(location=[lat,lon], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [color.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(final['Latitude'], final['Longitude'], final['Neighborhood'], final['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(Map2)
       
Map2

