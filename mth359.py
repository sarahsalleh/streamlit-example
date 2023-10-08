#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:13:13 2023

@author: drxav13r
"""

import requests
import pandas as pd
import numpy as np
from pandas import json_normalize
from scipy import interpolate
from scipy.stats import norm
import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
import time
import streamlit as st
# from BNBFuturesInterpolation import *
#start_time = time.time()

N = norm.cdf

def BSF(F, K, T, rd, sigma, call):
    phi = call + (call - 1)
    d1 = (np.log(F/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return phi * np.exp(-rd*T) * (F * N(phi*d1) - K *  N(phi*d2))

def BS_vanilla(S, K, T, rd, sigma, call):
    phi = call + (call - 1)
    d1 = (np.log(S/K) + (rd+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return phi  * (S * N(phi*d1) - K * np.exp(-rd*T) * N(phi*d2))

def vol_conversion(df):
    deltapctstr = ['10','25']
    for dpct in deltapctstr:
        df['call_'+dpct] = df['atm'] + df['bf'+dpct] + 0.5* df['rr'+dpct]
        df['put_'+dpct]  = df['atm'] + df['bf'+dpct] - 0.5* df['rr'+dpct]
    return df

def Delta(strike, rf, spot, tau, ivol, call):
    phi = call + (call - 1)
    d1 = (np.log(spot/strike) + (ivol**2/2)*tau) / (ivol*np.sqrt(tau))
    return phi * N(phi*d1) 

def fdelta_to_strike(F, pct_delta, rf, tau, volnp, call):
    # f = exp((rd-rf)*tau)
    phi = call + (call - 1)
    exponent = (-phi * norm.ppf( phi*  pct_delta ) * volnp * np.sqrt(tau) 
                + 0.5 * volnp * volnp * tau)
    strike = F * np.exp( exponent )
    return strike
    
def findVolgivenK(fwdcurve,option_vol_data,strike,duration):
    ivol = interpolate_vol(fwdcurve, 0.5, duration, option_vol_data)
    for i in range(1000):
        F = interpolate_future(duration,fwdcurve)

        # Calculate Delta based on your implementation
        delta = Delta(strike, 0, F, duration, ivol, isCall)  # Provide the correct value for isCall

        # Define your target Delta value (e.g., 0.5)
        target_delta = 0.5

        # Tolerance for Delta adjustment
        delta_tolerance = 0.01

        if abs(delta - target_delta) < delta_tolerance:
           
            break
        elif delta < target_delta:
            # If Delta is less than the target, increase ivol
            ivol += 0.001  # Adjust this increment as needed
        else:
            # If Delta is greater than the target, decrease ivol
            ivol -= 0.001  # Adjust this decrement as needed

    return ivol
    
_4PM = datetime.time(hour=16)
_FRI = 4 # Monday=0 for weekday()
 
def next_friday_4pm(now):
    if now.time() < _4PM:
        now = now.combine(now.date(),_4PM)
    else:
        now = now.combine(now.date(),_4PM) + datetime.timedelta(days=1)
    return now + datetime.timedelta((_FRI - now.weekday()) % 7)

def plot_3d(x, y, z, w=None, show=True):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return fig
    

# Get a list of all active BTC options from the Deribit API.
def get_all_active_options(ccy_str):
    import urllib.request, json
    url =  "https://deribit.com/api/v2/public/get_instruments?currency="+ccy_str+"&kind=option&expired=false"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = pd.DataFrame(data['result']).set_index('instrument_name')
    data['creation_date'] = pd.to_datetime(data['creation_timestamp'], unit='ms')
    data['expiration_date'] = pd.to_datetime(data['expiration_timestamp'], unit='ms')
    print(f'{data.shape[0]} active options.')
    return data

# Filter options based on data available from 'get_instruments'
def filter_options(price, active_options):
    # price is the current price of BTC
    
    #Get Put/Call information
    pc = active_options.index.str.strip().str[-1]

    # Set "moneyness"
    active_options['m'] = np.log(active_options['strike']/price)
    active_options.loc[pc=='P','m'] = -active_options['m']
    # Set days until expiration
    td = (active_options['expiration_date']-pd.Timestamp.today())
    active_options['t']= td.dt.seconds/(3600*24) + td.dt.days 
    # active_options['t'] = (active_options['expiration_date']-pd.Timestamp.today()).dt.days
    
    # Only include options that are less than 40% away from the current price and have less than 91 days until expiration
    active_options = active_options.query('m>=0 & m<.34 & t<366')
    
    print(f'{active_options.shape[0]} active options after filter.')
    return active_options

# Get Tick data for a given instrument from the Deribit API
def get_tick_data(instrument_name):
    import urllib.request, json
    url =  "https://deribit.com/api/v2/public/ticker?instrument_name="+instrument_name
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = json_normalize(data['result'])
    data.index = [instrument_name]
    return data

# Loop through all filtered options to get the current 'ticker' datas
@st.cache_resource(ttl=3600)
def get_all_option_data(ccy_str):
    instrument = ccy_str+"-PERPETUAL"
    option_data = get_tick_data(instrument)
    options = filter_options(option_data['last_price'][0], get_all_active_options(ccy_str))
    for o in options.index:
        option_data = pd.concat([option_data,get_tick_data(o)])
    option_data['call'] = option_data.index.str.strip().str[-1]=='C'
    #print(option_data)
    return option_data

@st.cache_resource(ttl=3600,experimental_allow_widgets=True)
def get_all_option_data_cache(ccy_str):
    #button = st.checkbox("Use pre-trained model:")
    option_data = get_all_option_data_cache(ccy_str)
    return option_data

# Interpolate Asset's future using forward-curve
def create_forwardcurve(asset):
    url = 'https://dapi.binance.com/dapi/v1/exchangeInfo'
    info = requests.get(url).json()['symbols']
    df = pd.DataFrame(info)
    df = df[(df['contractType']=='CURRENT_QUARTER') | (df['contractType']=='NEXT_QUARTER')]
    df = df[(df['baseAsset']) == asset]
    df = df[['symbol', 'deliveryDate']]
    df['type'] = 'inverse'
    # df = bnb_futures
    
    bid_list = []
    bid_size = []
    ask_list = []
    ask_size = []
    
    for x in df['symbol'].values:
        url = 'https://dapi.binance.com/dapi/v1/ticker/bookTicker?symbol='+x
        info = requests.get(url).json()[0]
        bid_list.append(float(info['bidPrice']))
        bid_size.append(float(info['bidQty']))
        ask_list.append(float(info['askPrice']))
        ask_size.append(float(info['askQty']))
    
    df['bid'] = bid_list
    df['bid_size'] = bid_size
    df['ask'] = ask_list
    df['ask_size'] = ask_size
    TTM = (df['deliveryDate']/1000) - (time.time())
    df['to_expiry'] = TTM
    df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], unit = 'ms')
    
    url = 'https://api.binance.com/api/v3/depth?limit=1&symbol='+asset+'USDT'
    info = requests.get(url).json()
    data = [asset+'USDT', '-', 'spot',
            float(info['bids'][0][0]), float(info['bids'][0][1]), float(info['asks'][0][0]), float(info['asks'][0][1]), float(0)]
    spot_df = pd.DataFrame(data).T
    spot_df.columns = ['symbol', 'deliveryDate', 'type', 'bid', 'bid_size', 'ask', 'ask_size', 'to_expiry']
    df = pd.concat([spot_df, df])
    df['to_expiry']=df['to_expiry']/3600/24
    
    x = df['to_expiry']
    y = 0.5*( df['ask'] + df['bid'] )
    cs = interpolate.CubicSpline(x, y)
    return cs

def interpolate_future(duration, curve):
    return curve(duration)

def load_realized_vols():
    coins = ['ETH', 'BNB']
    interval = '1h'
    end = round(time.time() * 1000)
    
    #spot
    all_data = []
    
    for x in coins:
        start = 1660000000000
        symbol = x
        data = []
        # count = 0
        
        while True:
            try:
                url = 'https://api.binance.com/api/v3/klines?symbol='+symbol+'USDT&interval='+interval+'&startTime='+str(start)+'&endTime='+str(end)
                info = requests.get(url).json()
                data.append(info)
                start = str(info[-1][0] + 3600000)  
                # count += 1
                # print(count)
            except:
                break
        
        data = [ele for innerlist in data for ele in innerlist]
        all_data.append(data)
        
    ethusdt_spot = pd.DataFrame(all_data[0], columns=['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    bnbusdt_spot = pd.DataFrame(all_data[1], columns=['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    
    ethusdt_spot.index = (pd.to_datetime(ethusdt_spot['close_time'], unit='ms'))
    ethusdt_spot = ethusdt_spot.drop(columns=['close_time', 'o','h','l','v','open_time','qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    ethusdt_spot = ethusdt_spot.resample('H').last()
    ethusdt_spot['lpx'] = np.log(ethusdt_spot['c'].astype(float))
    ethusdt_spot['lrets'] = ethusdt_spot['lpx'].diff()   
    
    bnbusdt_spot.index = (pd.to_datetime(bnbusdt_spot['close_time'], unit='ms'))
    bnbusdt_spot = bnbusdt_spot.drop(columns=['close_time', 'o','h','l','v','open_time','qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    bnbusdt_spot = bnbusdt_spot.resample('H').last()
    bnbusdt_spot['lpx'] = np.log(bnbusdt_spot['c'].astype(float))
    bnbusdt_spot['lrets'] = bnbusdt_spot['lpx'].diff()   
    
    ew = [7, 14, 30] * 24
    bnbvols = []
    ethvols = []
    for w in ew:
        bnbvolpd = bnbusdt_spot['lrets'].ewm(span=w).std() * (365*24)**0.5
        ethvolpd = ethusdt_spot['lrets'].ewm(span=w).std() * (365*24)**0.5
        # bnbvolpd = bnbusdt_spot['lrets'].rolling(w).std() * (365*24)**0.5
        # ethvolpd = ethusdt_spot['lrets'].rolling(w).std() * (365*24)**0.5
        bnbvols.append(bnbvolpd[-1])
        ethvols.append(ethvolpd[-1])
        
    bnbwvols = 0.3*bnbvols[0] + 0.4*bnbvols[1] + 0.5*bnbvols[2]
    ethvols = 0.3*ethvols[0] + 0.4*ethvols[1] + 0.5*ethvols[2]
    return max([bnbwvols-ethvols,0])
    

def adjustBNBsurface(volTableETH_df):
    """
    load_realized_vols(): This function presumably loads historical or realized volatility data for BNB and ETH, which are then used to calculate the volatility differential (BNBtoETHRVdiff).

    volTableBNB_df: The function creates a new DataFrame called volTableBNB_df by adding 75% of the volatility differential (BNBtoETHRVdiff) to the original volatility data in volTableETH_df. This effectively increases the volatility values in the BNB-related options within the surface.

    'ATM' Column: The function also adds a new column named 'ATM' to volTableBNB_df. This column is calculated by adding 25% of the volatility differential to the At-The-Money (ATM) volatility values in volTableETH_df. The ATM volatility is often considered the central volatility value on the volatility surface.

    Return: The adjusted volTableBNB_df DataFrame is returned as the result of the function.

    In summary, the adjustBNBsurface function adjusts the volatility surface for BNB options by increasing volatility values based on historical volatility differentials between BNB and ETH. This adjustment helps account for any observed differences in implied volatilities between the two assets.
    """
    BNBtoETHRVdiff = load_realized_vols()
    volTableBNB_df = volTableETH_df + 0.75 * BNBtoETHRVdiff
    volTableBNB_df['ATM'] = volTableETH_df['ATM'] + 0.25 * BNBtoETHRVdiff
    return volTableBNB_df
    

############ LOAD MARKET ETH DATA FROM DERIBIT ###########################
ccy_str = "ETH"
option_data = get_all_option_data(ccy_str)
option_vol_data = option_data.iloc[1:].copy()
### Add additional metrics to data
option_vol_data['t'] = np.nan; option_vol_data['strike'] = np.nan
# Calculated days until expiration
expDt = pd.to_datetime(option_vol_data.index.map(lambda x: x.split('-')[1])) + timedelta(hours=16)              
td = (expDt-pd.Timestamp.today())
option_vol_data['t'] = td.seconds/(3600*24) + td.days
# Pull strike from instrument name
option_vol_data['strike'] = option_vol_data.index.map(lambda x: x.split('-')[2]).astype(int)
option_vol_data = option_vol_data.rename(columns={'greeks.delta' : 'Delta', 'greeks.vega' : 'Vega' })
option_vol_data.to_csv('test.csv', index=False)

# Load ETH Futures and Spot
fwdcurveETH = create_forwardcurve('ETH')
a = requests.get("https://api.binance.com/api/v3/ticker/price").json()
dfpx = pd.DataFrame(a)
ethSpot = dfpx[dfpx['symbol']=='ETHUSDT']['price'].astype('float').iloc[0]


# Extract ETH Imp Vol surface 
def interpolate_vol(fwdcurve, deltak, duration, option_vol_data):
    df = option_vol_data[['mark_iv','t','call']].copy()
    if(np.array(option_vol_data['Delta']).size>0):
        df['Delta'] = option_vol_data['Delta'].copy()
    else:
        fpx = interpolate_future(option_vol_data['t'],fwdcurve)
        df['Delta'] = Delta(option_vol_data['strike'], 0, fpx, df['t'], df['mark_iv'], df['call']) 
    df.loc[df['Delta']<0,'Delta'] = df['Delta'] + 1
    df = df.sort_values(['t','Delta'],ascending = [True, False])
            
    # Interpolate 
    x = df['Delta']
    y = df['t']
    z = df['mark_iv']
    rbf3 = Rbf(x.values, y.values, z.values, function='linear', smooth=0.1)
    
    deltak = np.array(deltak)
    deltak[deltak<0] = deltak[deltak<0] + 1
    value = rbf3(deltak, duration)
    
    return value

def interpolate_volK(fwdcurve, strike, duration, option_vol_data):
    df = option_vol_data[['mark_iv','t','strike','call']].copy()
    df = df.sort_values(['t','strike'],ascending = [True, True])
            
    # Interpolate 
    x = df['strike']
    y = df['t']
    z = df['mark_iv']
    rbf3 = Rbf(x.values, y.values, z.values, function='linear', smooth=0.1)
    
    value = rbf3(strike, duration)
    
    return value

# Standardize unit 

option_vol_data['mark_iv'] = option_vol_data['mark_iv']/100
option_vol_data['t'] = option_vol_data['t']/365

# ETH Vol Surfaces For Display
tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
tenorsLabelMap = ['1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y']
deltaK = np.array([0.9, 0.75, 0.5, 0.25, 0.1])
deltaKLabelMap = ['10D Put','25D Put','ATM','25D Call','10D Call'] 
X,Y = np.meshgrid(deltaK,tenors)
volTableETH = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)

# # ETH Vol Surfaces For Display with switches
def buildETHvolTableView(fwdcurveETH,X,Y,option_vol_data, switchToExpDate=False, switchToStrike=False):
    tenorsLabelMap = ['1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y']
    tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
    if switchToExpDate==True:
        tenors = option_vol_data['t']  
        tenorsLabelMap = expDt.strftime('%Y-%m-%d').tolist()
    volTableETH = []
    if switchToStrike==True:
        deltaKLabelMap = option_vol_data['strike'].sort_values().unique().tolist()
        strikes = option_vol_data['strike'].sort_values().unique()
        X,Y = np.meshgrid(strikes,tenors)
        volTableETH = interpolate_volK(fwdcurveETH,X,Y,option_vol_data)
        volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)
    else:
        deltaKLabelMap = ['10D Put','25D Put','ATM','25D Call','10D Call'] 
        deltaK = np.array([0.9, 0.75, 0.5, 0.25, 0.1])
        X,Y = np.meshgrid(deltaK,tenors)
        volTableETH = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
        volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)       
    return volTableETH_df

# Plotting Vol Surface
yy = np.linspace(1,365)/365
xx = np.linspace(0.9,0.1)
X,Y = np.meshgrid(xx,yy)
Z = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
eth_plot = plot_3d(X, Y, Z)
 
#### Loading BNB Spot 
a = requests.get("https://api.binance.com/api/v3/ticker/price").json()
dfpx = pd.DataFrame(a)
bnbspot = dfpx[dfpx['symbol']=='BNBUSDT']['price'].astype('float').iloc[0]
fwdcurveBNB = create_forwardcurve('BNB')

### Building custom BNB vol (Initialize with ETH Vol Surface)
bnb_data = pd.DataFrame()

bnb_data['mark_iv'] = volTableETH.reshape(-1)
call = []
deltaK = []
len5 = int(len(bnb_data)/5)
tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
Texp = []
for i in range(len5):
    if i==0:
        call = [False, False, True, True, True]
    else:
        call = np.concatenate((call,[False, False, True, True, True]),dtype='bool' )
    deltaK = np.concatenate((deltaK,[0.9, 0.75, 0.5, 0.25, 0.1]))
    Texp = np.concatenate((Texp,tenors[i]*np.ones(5)))
    
bnb_data['Delta']=deltaK
bnb_data.loc[bnb_data['Delta']>0.5,'Delta'] = bnb_data['Delta'] - 1
bnb_data['call']=call
bnb_data['t']=Texp

#end_time = time.time()

#print("Time elapsed in this example code: ", end_time - start_time)

#######streamlit#######

if st.button("Refresh Vol surface"):
    # Clears all st.cache_resource caches:
    st.cache_resource.clear()
    st.experimental_rerun()

st.title('ETH Spot: :blue['+str(ethSpot)+']')
switchToExpDate = st.checkbox('Expiry Dates')
switchToStrike = st.checkbox('USDT Strikes')
volTableETH_view = buildETHvolTableView(fwdcurveETH,X,Y,option_vol_data, switchToExpDate, switchToStrike)
st.dataframe(volTableETH_view, use_container_width=True)
st.pyplot(eth_plot)

strike = st.number_input('Strike',value = 1800.0,step = 1.0,max_value=10000.0,min_value=1.0,format="%.2f")
duration = st.number_input('Time to Exp (Days)',value = 7.0,step = 0.01,max_value=365.0,min_value=0.01,format="%.2f")/365
isCall = st.checkbox('Call',value=True)
fpx = interpolate_future(duration,fwdcurveETH)
px_vol = findVolgivenK(fwdcurveETH,option_vol_data,strike,duration)
price = BS_vanilla(fpx, strike, duration, 0, px_vol, isCall )
deltak = Delta(strike,0,fpx,duration,px_vol,isCall)

table = pd.DataFrame({'Forward':fpx,'deltaK (%)':deltak*100,'strike':strike,'Days to Exp':duration*365,'Vol':px_vol,'Option Px (USDT)':price, 'Option Px (BNB)':price/bnbspot},index=['pricer'])
st.dataframe(table.style.format("{:.6}"))

st.title('BNB Spot: :blue['+str(bnbspot)+']')
volTableBNB_df = adjustBNBsurface(volTableETH_df)

if st.button("Reset BNB table"):
    st.session_state.exp_data_frame = st.experimental_data_editor(volTableBNB_df, use_container_width=True)
    edited_volTableBNB_df = st.session_state.exp_data_frame
    
elif 'exp_data_frame' not in st.session_state:
    st.session_state.exp_data_frame = st.experimental_data_editor(volTableBNB_df, use_container_width=True)
    edited_volTableBNB_df = st.session_state.exp_data_frame
    
else:
    edited_volTableBNB_df = st.experimental_data_editor(st.session_state.exp_data_frame, use_container_width=True) 
 
bnb_data['mark_iv'] = np.array(edited_volTableBNB_df).reshape(-1)

# Plotting Vol Surface
yy = np.linspace(1,365)/365
xx = np.linspace(0.9,0.1)
X,Y = np.meshgrid(xx,yy)
Z = interpolate_vol(fwdcurveBNB,X,Y,bnb_data)
bnb_plot = plot_3d(X, Y, Z)
st.pyplot(bnb_plot)

# deltak = st.number_input('DeltaK (+ call, - put)',value = 10.0,step = 0.1,max_value=100.0,min_value=-100.0,format="%.2f")/100
strikeb = st.number_input('Strike',value = 350.0,step = 1.0,max_value=700.0,min_value=0.2,format="%.2f")
durationb = st.number_input('Time to Exp (Days)',value = 7.0,step = 0.01,max_value=365.0,min_value=0.02,format="%.2f")/365
isCall = st.checkbox('Call',value=True, key='bnbcall')
fbnb = interpolate_future(durationb,fwdcurveBNB)
b_vol = findVolgivenK(fwdcurveBNB,bnb_data,strikeb,durationb)
priceb = BS_vanilla(fbnb, strikeb, durationb, 0, b_vol, isCall )
deltak = Delta(strikeb,0,fbnb,durationb,b_vol,isCall)

table = pd.DataFrame({'Forward':fbnb,'deltaK (%)':deltak*100,'strike':strikeb,'Days to Exp':durationb*365,'Vol':b_vol,'Option Px (USDT)':priceb, 'Option Px (BNB)':priceb/bnbspot},index=['pricer'])
st.dataframe(table.style.format("{:.5}"))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:13:13 2023

@author: drxav13r
"""

import requests
import pandas as pd
import numpy as np
from pandas import json_normalize
from scipy import interpolate
from scipy.stats import norm
import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
import time
import streamlit as st
# from BNBFuturesInterpolation import *
#start_time = time.time()

N = norm.cdf

def BSF(F, K, T, rd, sigma, call):
    phi = call + (call - 1)
    d1 = (np.log(F/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return phi * np.exp(-rd*T) * (F * N(phi*d1) - K *  N(phi*d2))

def BS_vanilla(S, K, T, rd, sigma, call):
    phi = call + (call - 1)
    d1 = (np.log(S/K) + (rd+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return phi  * (S * N(phi*d1) - K * np.exp(-rd*T) * N(phi*d2))

def vol_conversion(df):
    deltapctstr = ['10','25']
    for dpct in deltapctstr:
        df['call_'+dpct] = df['atm'] + df['bf'+dpct] + 0.5* df['rr'+dpct]
        df['put_'+dpct]  = df['atm'] + df['bf'+dpct] - 0.5* df['rr'+dpct]
    return df

def Delta(strike, rf, spot, tau, ivol, call):
    phi = call + (call - 1)
    d1 = (np.log(spot/strike) + (ivol**2/2)*tau) / (ivol*np.sqrt(tau))
    return phi * N(phi*d1) 

def fdelta_to_strike(F, pct_delta, rf, tau, volnp, call):
    # f = exp((rd-rf)*tau)
    phi = call + (call - 1)
    exponent = (-phi * norm.ppf( phi*  pct_delta ) * volnp * np.sqrt(tau) 
                + 0.5 * volnp * volnp * tau)
    strike = F * np.exp( exponent )
    return strike
    
def findVolgivenK(fwdcurve,option_vol_data,strike,duration):
    ivol = interpolate_vol(fwdcurve, 0.5, duration, option_vol_data)
    for i in range(1000):
        F = interpolate_future(duration,fwdcurve)

        # Calculate Delta based on your implementation
        delta = Delta(strike, 0, F, duration, ivol, isCall)  # Provide the correct value for isCall

        # Define your target Delta value (e.g., 0.5)
        target_delta = 0.5

        # Tolerance for Delta adjustment
        delta_tolerance = 0.01

        if abs(delta - target_delta) < delta_tolerance:
           
            break
        elif delta < target_delta:
            # If Delta is less than the target, increase ivol
            ivol += 0.001  # Adjust this increment as needed
        else:
            # If Delta is greater than the target, decrease ivol
            ivol -= 0.001  # Adjust this decrement as needed

    return ivol
    
_4PM = datetime.time(hour=16)
_FRI = 4 # Monday=0 for weekday()
 
def next_friday_4pm(now):
    if now.time() < _4PM:
        now = now.combine(now.date(),_4PM)
    else:
        now = now.combine(now.date(),_4PM) + datetime.timedelta(days=1)
    return now + datetime.timedelta((_FRI - now.weekday()) % 7)

def plot_3d(x, y, z, w=None, show=True):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return fig
    

# Get a list of all active BTC options from the Deribit API.
def get_all_active_options(ccy_str):
    import urllib.request, json
    url =  "https://deribit.com/api/v2/public/get_instruments?currency="+ccy_str+"&kind=option&expired=false"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = pd.DataFrame(data['result']).set_index('instrument_name')
    data['creation_date'] = pd.to_datetime(data['creation_timestamp'], unit='ms')
    data['expiration_date'] = pd.to_datetime(data['expiration_timestamp'], unit='ms')
    print(f'{data.shape[0]} active options.')
    return data

# Filter options based on data available from 'get_instruments'
def filter_options(price, active_options):
    # price is the current price of BTC
    
    #Get Put/Call information
    pc = active_options.index.str.strip().str[-1]

    # Set "moneyness"
    active_options['m'] = np.log(active_options['strike']/price)
    active_options.loc[pc=='P','m'] = -active_options['m']
    # Set days until expiration
    td = (active_options['expiration_date']-pd.Timestamp.today())
    active_options['t']= td.dt.seconds/(3600*24) + td.dt.days 
    # active_options['t'] = (active_options['expiration_date']-pd.Timestamp.today()).dt.days
    
    # Only include options that are less than 40% away from the current price and have less than 91 days until expiration
    active_options = active_options.query('m>=0 & m<.34 & t<366')
    
    print(f'{active_options.shape[0]} active options after filter.')
    return active_options

# Get Tick data for a given instrument from the Deribit API
def get_tick_data(instrument_name):
    import urllib.request, json
    url =  "https://deribit.com/api/v2/public/ticker?instrument_name="+instrument_name
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = json_normalize(data['result'])
    data.index = [instrument_name]
    return data

# Loop through all filtered options to get the current 'ticker' datas
@st.cache_resource(ttl=3600)
def get_all_option_data(ccy_str):
    instrument = ccy_str+"-PERPETUAL"
    option_data = get_tick_data(instrument)
    options = filter_options(option_data['last_price'][0], get_all_active_options(ccy_str))
    for o in options.index:
        option_data = pd.concat([option_data,get_tick_data(o)])
    option_data['call'] = option_data.index.str.strip().str[-1]=='C'
    #print(option_data)
    return option_data

@st.cache_resource(ttl=3600,experimental_allow_widgets=True)
def get_all_option_data_cache(ccy_str):
    #button = st.checkbox("Use pre-trained model:")
    option_data = get_all_option_data_cache(ccy_str)
    return option_data

# Interpolate Asset's future using forward-curve
def create_forwardcurve(asset):
    url = 'https://dapi.binance.com/dapi/v1/exchangeInfo'
    info = requests.get(url).json()['symbols']
    df = pd.DataFrame(info)
    df = df[(df['contractType']=='CURRENT_QUARTER') | (df['contractType']=='NEXT_QUARTER')]
    df = df[(df['baseAsset']) == asset]
    df = df[['symbol', 'deliveryDate']]
    df['type'] = 'inverse'
    # df = bnb_futures
    
    bid_list = []
    bid_size = []
    ask_list = []
    ask_size = []
    
    for x in df['symbol'].values:
        url = 'https://dapi.binance.com/dapi/v1/ticker/bookTicker?symbol='+x
        info = requests.get(url).json()[0]
        bid_list.append(float(info['bidPrice']))
        bid_size.append(float(info['bidQty']))
        ask_list.append(float(info['askPrice']))
        ask_size.append(float(info['askQty']))
    
    df['bid'] = bid_list
    df['bid_size'] = bid_size
    df['ask'] = ask_list
    df['ask_size'] = ask_size
    TTM = (df['deliveryDate']/1000) - (time.time())
    df['to_expiry'] = TTM
    df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], unit = 'ms')
    
    url = 'https://api.binance.com/api/v3/depth?limit=1&symbol='+asset+'USDT'
    info = requests.get(url).json()
    data = [asset+'USDT', '-', 'spot',
            float(info['bids'][0][0]), float(info['bids'][0][1]), float(info['asks'][0][0]), float(info['asks'][0][1]), float(0)]
    spot_df = pd.DataFrame(data).T
    spot_df.columns = ['symbol', 'deliveryDate', 'type', 'bid', 'bid_size', 'ask', 'ask_size', 'to_expiry']
    df = pd.concat([spot_df, df])
    df['to_expiry']=df['to_expiry']/3600/24
    
    x = df['to_expiry']
    y = 0.5*( df['ask'] + df['bid'] )
    cs = interpolate.CubicSpline(x, y)
    return cs

def interpolate_future(duration, curve):
    return curve(duration)

def load_realized_vols():
    coins = ['ETH', 'BNB']
    interval = '1h'
    end = round(time.time() * 1000)
    
    #spot
    all_data = []
    
    for x in coins:
        start = 1660000000000
        symbol = x
        data = []
        # count = 0
        
        while True:
            try:
                url = 'https://api.binance.com/api/v3/klines?symbol='+symbol+'USDT&interval='+interval+'&startTime='+str(start)+'&endTime='+str(end)
                info = requests.get(url).json()
                data.append(info)
                start = str(info[-1][0] + 3600000)  
                # count += 1
                # print(count)
            except:
                break
        
        data = [ele for innerlist in data for ele in innerlist]
        all_data.append(data)
        
    ethusdt_spot = pd.DataFrame(all_data[0], columns=['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    bnbusdt_spot = pd.DataFrame(all_data[1], columns=['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    
    ethusdt_spot.index = (pd.to_datetime(ethusdt_spot['close_time'], unit='ms'))
    ethusdt_spot = ethusdt_spot.drop(columns=['close_time', 'o','h','l','v','open_time','qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    ethusdt_spot = ethusdt_spot.resample('H').last()
    ethusdt_spot['lpx'] = np.log(ethusdt_spot['c'].astype(float))
    ethusdt_spot['lrets'] = ethusdt_spot['lpx'].diff()   
    
    bnbusdt_spot.index = (pd.to_datetime(bnbusdt_spot['close_time'], unit='ms'))
    bnbusdt_spot = bnbusdt_spot.drop(columns=['close_time', 'o','h','l','v','open_time','qav', 'no_of_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    bnbusdt_spot = bnbusdt_spot.resample('H').last()
    bnbusdt_spot['lpx'] = np.log(bnbusdt_spot['c'].astype(float))
    bnbusdt_spot['lrets'] = bnbusdt_spot['lpx'].diff()   
    
    ew = [7, 14, 30] * 24
    bnbvols = []
    ethvols = []
    for w in ew:
        bnbvolpd = bnbusdt_spot['lrets'].ewm(span=w).std() * (365*24)**0.5
        ethvolpd = ethusdt_spot['lrets'].ewm(span=w).std() * (365*24)**0.5
        # bnbvolpd = bnbusdt_spot['lrets'].rolling(w).std() * (365*24)**0.5
        # ethvolpd = ethusdt_spot['lrets'].rolling(w).std() * (365*24)**0.5
        bnbvols.append(bnbvolpd[-1])
        ethvols.append(ethvolpd[-1])
        
    bnbwvols = 0.3*bnbvols[0] + 0.4*bnbvols[1] + 0.5*bnbvols[2]
    ethvols = 0.3*ethvols[0] + 0.4*ethvols[1] + 0.5*ethvols[2]
    return max([bnbwvols-ethvols,0])
    

def adjustBNBsurface(volTableETH_df):
    """
    load_realized_vols(): This function presumably loads historical or realized volatility data for BNB and ETH, which are then used to calculate the volatility differential (BNBtoETHRVdiff).

    volTableBNB_df: The function creates a new DataFrame called volTableBNB_df by adding 75% of the volatility differential (BNBtoETHRVdiff) to the original volatility data in volTableETH_df. This effectively increases the volatility values in the BNB-related options within the surface.

    'ATM' Column: The function also adds a new column named 'ATM' to volTableBNB_df. This column is calculated by adding 25% of the volatility differential to the At-The-Money (ATM) volatility values in volTableETH_df. The ATM volatility is often considered the central volatility value on the volatility surface.

    Return: The adjusted volTableBNB_df DataFrame is returned as the result of the function.

    In summary, the adjustBNBsurface function adjusts the volatility surface for BNB options by increasing volatility values based on historical volatility differentials between BNB and ETH. This adjustment helps account for any observed differences in implied volatilities between the two assets.
    """
    BNBtoETHRVdiff = load_realized_vols()
    volTableBNB_df = volTableETH_df + 0.75 * BNBtoETHRVdiff
    volTableBNB_df['ATM'] = volTableETH_df['ATM'] + 0.25 * BNBtoETHRVdiff
    return volTableBNB_df
    

############ LOAD MARKET ETH DATA FROM DERIBIT ###########################
ccy_str = "ETH"
option_data = get_all_option_data(ccy_str)
option_vol_data = option_data.iloc[1:].copy()
### Add additional metrics to data
option_vol_data['t'] = np.nan; option_vol_data['strike'] = np.nan
# Calculated days until expiration
expDt = pd.to_datetime(option_vol_data.index.map(lambda x: x.split('-')[1])) + timedelta(hours=16)              
td = (expDt-pd.Timestamp.today())
option_vol_data['t'] = td.seconds/(3600*24) + td.days
# Pull strike from instrument name
option_vol_data['strike'] = option_vol_data.index.map(lambda x: x.split('-')[2]).astype(int)
option_vol_data = option_vol_data.rename(columns={'greeks.delta' : 'Delta', 'greeks.vega' : 'Vega' })
option_vol_data.to_csv('test.csv', index=False)

# Load ETH Futures and Spot
fwdcurveETH = create_forwardcurve('ETH')
a = requests.get("https://api.binance.com/api/v3/ticker/price").json()
dfpx = pd.DataFrame(a)
ethSpot = dfpx[dfpx['symbol']=='ETHUSDT']['price'].astype('float').iloc[0]


# Extract ETH Imp Vol surface 
def interpolate_vol(fwdcurve, deltak, duration, option_vol_data):
    df = option_vol_data[['mark_iv','t','call']].copy()
    if(np.array(option_vol_data['Delta']).size>0):
        df['Delta'] = option_vol_data['Delta'].copy()
    else:
        fpx = interpolate_future(option_vol_data['t'],fwdcurve)
        df['Delta'] = Delta(option_vol_data['strike'], 0, fpx, df['t'], df['mark_iv'], df['call']) 
    df.loc[df['Delta']<0,'Delta'] = df['Delta'] + 1
    df = df.sort_values(['t','Delta'],ascending = [True, False])
            
    # Interpolate 
    x = df['Delta']
    y = df['t']
    z = df['mark_iv']
    rbf3 = Rbf(x.values, y.values, z.values, function='linear', smooth=0.1)
    
    deltak = np.array(deltak)
    deltak[deltak<0] = deltak[deltak<0] + 1
    value = rbf3(deltak, duration)
    
    return value

def interpolate_volK(fwdcurve, strike, duration, option_vol_data):
    df = option_vol_data[['mark_iv','t','strike','call']].copy()
    df = df.sort_values(['t','strike'],ascending = [True, True])
            
    # Interpolate 
    x = df['strike']
    y = df['t']
    z = df['mark_iv']
    rbf3 = Rbf(x.values, y.values, z.values, function='linear', smooth=0.1)
    
    value = rbf3(strike, duration)
    
    return value

# Standardize unit 

option_vol_data['mark_iv'] = option_vol_data['mark_iv']/100
option_vol_data['t'] = option_vol_data['t']/365

# ETH Vol Surfaces For Display
tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
tenorsLabelMap = ['1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y']
deltaK = np.array([0.9, 0.75, 0.5, 0.25, 0.1])
deltaKLabelMap = ['10D Put','25D Put','ATM','25D Call','10D Call'] 
X,Y = np.meshgrid(deltaK,tenors)
volTableETH = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)

# # ETH Vol Surfaces For Display with switches
def buildETHvolTableView(fwdcurveETH,X,Y,option_vol_data, switchToExpDate=False, switchToStrike=False):
    tenorsLabelMap = ['1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y']
    tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
    if switchToExpDate==True:
        tenors = option_vol_data['t']  
        tenorsLabelMap = expDt.strftime('%Y-%m-%d').tolist()
    volTableETH = []
    if switchToStrike==True:
        deltaKLabelMap = option_vol_data['strike'].sort_values().unique().tolist()
        strikes = option_vol_data['strike'].sort_values().unique()
        X,Y = np.meshgrid(strikes,tenors)
        volTableETH = interpolate_volK(fwdcurveETH,X,Y,option_vol_data)
        volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)
    else:
        deltaKLabelMap = ['10D Put','25D Put','ATM','25D Call','10D Call'] 
        deltaK = np.array([0.9, 0.75, 0.5, 0.25, 0.1])
        X,Y = np.meshgrid(deltaK,tenors)
        volTableETH = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
        volTableETH_df = pd.DataFrame(volTableETH,index = tenorsLabelMap,columns = deltaKLabelMap)       
    return volTableETH_df

# Plotting Vol Surface
yy = np.linspace(1,365)/365
xx = np.linspace(0.9,0.1)
X,Y = np.meshgrid(xx,yy)
Z = interpolate_vol(fwdcurveETH,X,Y,option_vol_data)
eth_plot = plot_3d(X, Y, Z)
 
#### Loading BNB Spot 
a = requests.get("https://api.binance.com/api/v3/ticker/price").json()
dfpx = pd.DataFrame(a)
bnbspot = dfpx[dfpx['symbol']=='BNBUSDT']['price'].astype('float').iloc[0]
fwdcurveBNB = create_forwardcurve('BNB')

### Building custom BNB vol (Initialize with ETH Vol Surface)
bnb_data = pd.DataFrame()

bnb_data['mark_iv'] = volTableETH.reshape(-1)
call = []
deltaK = []
len5 = int(len(bnb_data)/5)
tenors = np.array([1,7,14,21,30,61,91,183,276,365])/365
Texp = []
for i in range(len5):
    if i==0:
        call = [False, False, True, True, True]
    else:
        call = np.concatenate((call,[False, False, True, True, True]),dtype='bool' )
    deltaK = np.concatenate((deltaK,[0.9, 0.75, 0.5, 0.25, 0.1]))
    Texp = np.concatenate((Texp,tenors[i]*np.ones(5)))
    
bnb_data['Delta']=deltaK
bnb_data.loc[bnb_data['Delta']>0.5,'Delta'] = bnb_data['Delta'] - 1
bnb_data['call']=call
bnb_data['t']=Texp

#end_time = time.time()

#print("Time elapsed in this example code: ", end_time - start_time)

#######streamlit#######

if st.button("Refresh Vol surface"):
    # Clears all st.cache_resource caches:
    st.cache_resource.clear()
    st.experimental_rerun()

st.title('ETH Spot: :blue['+str(ethSpot)+']')
switchToExpDate = st.checkbox('Expiry Dates')
switchToStrike = st.checkbox('USDT Strikes')
volTableETH_view = buildETHvolTableView(fwdcurveETH,X,Y,option_vol_data, switchToExpDate, switchToStrike)
st.dataframe(volTableETH_view, use_container_width=True)
st.pyplot(eth_plot)

strike = st.number_input('Strike',value = 1800.0,step = 1.0,max_value=10000.0,min_value=1.0,format="%.2f")
duration = st.number_input('Time to Exp (Days)',value = 7.0,step = 0.01,max_value=365.0,min_value=0.01,format="%.2f")/365
isCall = st.checkbox('Call',value=True)
fpx = interpolate_future(duration,fwdcurveETH)
px_vol = findVolgivenK(fwdcurveETH,option_vol_data,strike,duration)
price = BS_vanilla(fpx, strike, duration, 0, px_vol, isCall )
deltak = Delta(strike,0,fpx,duration,px_vol,isCall)

table = pd.DataFrame({'Forward':fpx,'deltaK (%)':deltak*100,'strike':strike,'Days to Exp':duration*365,'Vol':px_vol,'Option Px (USDT)':price, 'Option Px (BNB)':price/bnbspot},index=['pricer'])
st.dataframe(table.style.format("{:.6}"))

st.title('BNB Spot: :blue['+str(bnbspot)+']')
volTableBNB_df = adjustBNBsurface(volTableETH_df)

if st.button("Reset BNB table"):
    st.session_state.exp_data_frame = st.experimental_data_editor(volTableBNB_df, use_container_width=True)
    edited_volTableBNB_df = st.session_state.exp_data_frame
    
elif 'exp_data_frame' not in st.session_state:
    st.session_state.exp_data_frame = st.experimental_data_editor(volTableBNB_df, use_container_width=True)
    edited_volTableBNB_df = st.session_state.exp_data_frame
    
else:
    edited_volTableBNB_df = st.experimental_data_editor(st.session_state.exp_data_frame, use_container_width=True) 
 
bnb_data['mark_iv'] = np.array(edited_volTableBNB_df).reshape(-1)

# Plotting Vol Surface
yy = np.linspace(1,365)/365
xx = np.linspace(0.9,0.1)
X,Y = np.meshgrid(xx,yy)
Z = interpolate_vol(fwdcurveBNB,X,Y,bnb_data)
bnb_plot = plot_3d(X, Y, Z)
st.pyplot(bnb_plot)

# deltak = st.number_input('DeltaK (+ call, - put)',value = 10.0,step = 0.1,max_value=100.0,min_value=-100.0,format="%.2f")/100
strikeb = st.number_input('Strike',value = 350.0,step = 1.0,max_value=700.0,min_value=0.2,format="%.2f")
durationb = st.number_input('Time to Exp (Days)',value = 7.0,step = 0.01,max_value=365.0,min_value=0.02,format="%.2f")/365
isCall = st.checkbox('Call',value=True, key='bnbcall')
fbnb = interpolate_future(durationb,fwdcurveBNB)
b_vol = findVolgivenK(fwdcurveBNB,bnb_data,strikeb,durationb)
priceb = BS_vanilla(fbnb, strikeb, durationb, 0, b_vol, isCall )
deltak = Delta(strikeb,0,fbnb,durationb,b_vol,isCall)

table = pd.DataFrame({'Forward':fbnb,'deltaK (%)':deltak*100,'strike':strikeb,'Days to Exp':durationb*365,'Vol':b_vol,'Option Px (USDT)':priceb, 'Option Px (BNB)':priceb/bnbspot},index=['pricer'])
st.dataframe(table.style.format("{:.5}"))



