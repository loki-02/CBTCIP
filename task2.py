import pandas as pd
import matplotlib.pyplot as plt
unemployment_data = pd.read_excel('Unemployment in India.xlsx')
unemployment_data['Region'] = unemployment_data['Region'].astype(str)
plt.figure(figsize=(12, 10))
plt.plot(unemployment_data['Region'], unemployment_data['Estimated Unemployment Rate (%)'], marker='o', linestyle='-')
plt.title('Region wise unemployment')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.xticks(rotation=45)  
plt.tight_layout() 
plt.show()
