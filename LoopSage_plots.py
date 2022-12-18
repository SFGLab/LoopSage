import cairo
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def draw_contact(c, y, start, end, lw=0.002, h=0.4):
    c.set_source_rgba(0.0, 1.0, 0.0, 0.1)
    c.set_line_width(lw)
    c.move_to(start, y)
    length = end-start
    h = y-length*4/3
    c.curve_to(start+length/3, h, start+2*length/3, h, end, y)
    c.stroke()
#     c.arc_negative((start+end)/2, 0.9, length/2, 0, math.pi)
#     c.stroke()

def draw_arcplot(ms,ns,N_beads,lw_axis = 0.005,axis_h = 0.8):
    with cairo.SVGSurface("arcplot.svg", 800, 400) as surface:
        c = cairo.Context(surface)
        c.scale(800, 400)
        c.rectangle(0, 0, 800, 400)
        c.set_source_rgb(1, 1, 1)
        c.fill()

        for i in range(len(ms)):
            start = ms[i]
            end = ns[i]
            start = 0.1 + 0.8*start/N_beads
            end = 0.1 + 0.8*end/N_beads
            draw_contact(c=c, y=axis_h, start=start, end=end,
                        lw=0.002, h=0.4)

        # axis
        c.set_source_rgb(0.0, 0.0, 0.0)
        c.set_line_width(lw_axis)
        c.move_to(0.1, axis_h)
        c.line_to(0.9, axis_h)
        c.stroke()
        c.set_line_width(lw_axis/2)
        c.move_to(0.1, axis_h+0.01)
        c.line_to(0.1, axis_h-0.01)
        c.stroke()
        c.move_to(0.9, axis_h+0.01)
        c.line_to(0.9, axis_h-0.01)
        c.stroke()

def make_timeplots(Es, Bs, Ks, Fs):
    figure(figsize=(10, 8), dpi=80)
    plt.plot(Es,'k')
    plt.plot(Bs,'blue')
    plt.plot(Ks,'green')
    plt.plot(Fs,'red')
    plt.ylabel('Metrics',fontsize=16)
    plt.xlabel('Monte Carlo Step',fontsize=16)
    plt.yscale('symlog')
    plt.legend(['Total Energy','Binding','Knotting','Folding'],fontsize=16)
    plt.grid()
    plt.show()
