from Wave2D import *
import matplotlib.animation as animation

N0 = 2**(5)
Nt = 100

wave = Wave2D()
xij, yij, data = wave(N0, Nt, cfl=1/np.sqrt(2), mx=2, my=2, store_data=1)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
frames = []
for n, val in data.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
    #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(), 
    #                        vmax=data[0].max(), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    frames.append([frame])
    
ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                repeat_delay=1000)
ani.save('neumannwave.gif', writer='pillow', fps=5) # This animated png opens in a browser