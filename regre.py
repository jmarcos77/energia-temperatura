# Regressão Linear por cidade
for city in cities:
    df_city = df_all[df_all['city'] == city]
    X = df_city[['temperature']]
    y = df_city['consumption']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='temperature', y='consumption', data=df_city, alpha=0.3)
    sns.lineplot(x=df_city['temperature'], y=y_pred, color='red', label='Regressão Linear')
    plt.title(f'{city}: Consumo vs Temperatura\nR²={r2:.2f} | RMSE={rmse:.2f}')
    plt.xlabel('Temperatura Média (°C)')
    plt.ylabel('Consumo de Energia (kWh)')
    plt.tight_layout()
    plt.show()