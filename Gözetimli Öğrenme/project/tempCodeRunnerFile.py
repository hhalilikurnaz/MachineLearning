
def plot_hist(variable):
      plt.figure(figsize=(9,3))
      plt.hist(df[variable],bins=50)
      plt.xlabel(variable)
      plt.ylabel("Frequency")
      plt.title("{} distribution with hist".format(variable))
      plt.show() #==>>> histogram function

numeriVar=["Fare","Age"]
for n in numeriVar:
      plot_hist(n)