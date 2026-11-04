import cv2, numpy as np, math
exec(open(__file__.replace('sb_mock2.py','sb_mock.py')).read().split("n=70;")[0])   # reuse helpers + fall path
ghosts = ghosts[1:]                      # first copy sits on the release point
rng = np.random.default_rng(12)
TX0, TY1 = 13 + 1125*0.8/2, 67 + 1033*0.8 - 10          # tree centre x, base y
CYAN, AMBER, WHITE = (1.0,0.85,0.45), (0.35,0.75,1.0), (1.0,0.95,0.9)
FIRE = [dict(x=rng.uniform(40,900), y=rng.uniform(90,820), sp=rng.uniform(8,22), ph=rng.uniform(0,6.3),
             r=rng.uniform(1.4,3.2), c=CYAN if rng.random()<0.5 else AMBER) for _ in range(48)]
LEAVES = [dict(x=rng.uniform(150,800), y0=rng.uniform(250,600), t0=rng.uniform(-6,19), sp=rng.uniform(35,60),
               ph=rng.uniform(0,6.3)) for _ in range(9)]
def glow_draw(G, draw, amp=1.0, s=3):
    m=np.zeros((H,W),np.uint8); draw(m); f=m.astype(np.float32)/255
    return f, cv2.GaussianBlur(f,(0,0),s)
def floor(G, amp=1.0):
    m=np.zeros((H,W),np.uint8)
    for rx,ry,th in ((250,26,2),(390,42,2),(540,58,1)):
        cv2.ellipse(m,(int(TX0),int(TY1)),(rx,ry),0,0,360,255,th,cv2.LINE_AA)
    for k in range(16):                                   # dotted spokes on the floor
        a=k*math.pi/8
        for u in np.linspace(0.18,1.0,14):
            x=TX0+540*u*math.cos(a); y=TY1+58*u*math.sin(a)
            cv2.circle(m,(int(x),int(y)),1,200,-1,cv2.LINE_AA)
    f=m.astype(np.float32)/255
    G+=(f*0.55+cv2.GaussianBlur(f,(0,0),5)*0.9)[...,None]*np.array(CYAN,np.float32)*amp
def fireflies(G, t, amp=1.0, frozen=False):
    m=np.zeros((H,W),np.float32)
    for f in FIRE:
        tt = 13.7 if frozen and t>13.7 else t
        y=f['y']-f['sp']*tt; y=90+((y-90)%730); x=f['x']+14*math.sin(tt*0.6+f['ph'])
        tw=0.55+0.45*math.sin(tt*3+f['ph'])
        tmp=np.zeros((H,W),np.uint8); cv2.circle(tmp,(int(x),int(y)),int(f['r']+0.5),255,-1,cv2.LINE_AA)
        G+=(cv2.GaussianBlur(tmp.astype(np.float32)/255,(0,0),2.5)*1.6*tw*amp)[...,None]*np.array(f['c'],np.float32)
def leaves(G, t, amp=1.0, frozen=False):
    for L in LEAVES:
        tt=(13.7 if frozen and t>13.7 else t)-L['t0']
        if tt<0: continue
        y=L['y0']+L['sp']*tt; x=L['x']+40*math.sin(tt*1.3+L['ph'])
        if y>TY1-20: continue
        ang=math.degrees(math.sin(tt*1.7+L['ph']))*0.6+L['ph']*30
        m=np.zeros((H,W),np.uint8)
        pts=cv2.ellipse2Poly((int(x),int(y)),(16,6),int(ang),0,360,20)
        cv2.polylines(m,[pts],True,255,2,cv2.LINE_AA)
        a=math.radians(ang); cv2.line(m,(int(x-15*math.cos(a)),int(y-15*math.sin(a))),(int(x+15*math.cos(a)),int(y+15*math.sin(a))),180,1,cv2.LINE_AA)
        f=m.astype(np.float32)/255
        G+=(f*0.8+cv2.GaussianBlur(f,(0,0),3)*0.8)[...,None]*np.array((0.45,1.0,0.55),np.float32)*amp
def dotted_v(m,x,ya,yb,step=11,r=1):
    for y in np.arange(ya,yb,step): cv2.circle(m,(int(x),int(y)),r,255,-1,cv2.LINE_AA)
def dashed_h(m,xa,xb,y,dash=9,gap=7,th=1):
    x=xa
    while x<xb: cv2.line(m,(int(x),int(y)),(int(min(x+dash,xb)),int(y)),255,th,cv2.LINE_AA); x+=dash+gap
def scale(G, upto_y, amp=1.0):
    """Premium dotted measuring scale beside the fall path."""
    X=int(x0-120); ys=[y0]+[g for g in ghosts if g<=upto_y+1]+([upto_y] if upto_y>ghosts[-1]+1 or upto_y not in ghosts else [])
    ys=sorted(set([round(v,1) for v in ys]))
    m=np.zeros((H,W),np.uint8); lab=np.zeros((H,W),np.uint8)
    dotted_v(m,X,y0,ys[-1]+1)
    for y in ys:
        dashed_h(m,X-12,x0-34,y)
        cv2.circle(m,(X,int(y)),4,255,-1,cv2.LINE_AA)
    k = 0
    for a_,b_ in zip(ys[:-1],ys[1:]):
        if b_-a_<14: continue
        k += 1
        cv2.putText(lab,'d%d'%k,(X-44,int((a_+b_)/2)+6),cv2.FONT_HERSHEY_DUPLEX,0.55,255,1,cv2.LINE_AA)
    f=m.astype(np.float32)/255; l=lab.astype(np.float32)/255
    G+=(f*0.75+cv2.GaussianBlur(f,(0,0),3)*0.7)[...,None]*np.array(CYAN,np.float32)*amp
    G+=(l*0.95+cv2.GaussianBlur(l,(0,0),3)*0.5)[...,None]*np.array(AMBER,np.float32)*amp
def lively(G,t,frozen=False,amp=1.0):
    floor(G,amp); fireflies(G,t,amp,frozen); leaves(G,t,amp,frozen)

LANE = np.clip(1 - np.abs(xx - (x0 - 45)) / 190, 0, 1) ** 0.8 * np.clip((yy - (y0 - 90)) / 60, 0, 1)
LANE = cv2.GaussianBlur(LANE.astype(np.float32), (0,0), 25)
gray = APPLE.max(2, keepdims=True)
GHOST = gray * np.array([1.0, 0.92, 0.7], np.float32)          # white-cyan silhouette of the apple
def plate_lane(n, sat=1.0, tree_amp=0.93, lane=0.78):
    a=cv2.imread('masks_rvm/%05d.png'%(n+1),0).astype(np.float32)/255; fg=cv2.imread('fgr_rvm/%05d.png'%(n+1)).astype(np.float32)/255
    PG=(1-np.exp(-2*fg*0.95))/(1-math.exp(-2))*KEY[...,None]*np.array([1.06,0.98,0.92],np.float32)
    rgb=TR[...,:3]
    if sat<1: g=rgb.mean(2,keepdims=True); rgb=rgb*sat+g*(1-sat)*np.array([1.1,1.0,0.85])
    ta=TR[...,3:4]*tree_amp*(1-lane*LANE[...,None])
    return PG*a[...,None]+(NAVY*(1-ta)+rgb*ta)*(1-a[...,None])
def save(name,img): cv2.imwrite('sb/'+name,(np.clip(img,0,1)*255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,93])
# 1-3, 5: real frames from the test render + the new living layers
for n,name,amp in ((12,'p1.jpg',0.5),(19,'p2.jpg',1.0),(33,'p3.jpg',1.0),(108,'p5.jpg',1.0)):
    base=cv2.imread('sb/real_%d.jpg'%n).astype(np.float32)/255; G=np.zeros((H,W,3),np.float32); lively(G,n/25,amp=amp)
    save(name,fin(base,G))
# 4: time slow
n=70; G=np.zeros((H,W,3),np.float32); hp=P[n,0]; fp=P[n,1]; lively(G,n/25)
add(G,APPLE,hp[0]+4,hp[1]-16,72); add(G,KNOB,fp[0]+5,fp[1]-55,190,1.0,rot=40); add(G,SW,1690,300,400)
ripple(G,fp[0],fp[1]-55,260,(1.0,0.75,0.3),0.9); ripple(G,fp[0],fp[1]-55,420,(1.0,0.75,0.3),0.45)
save('p4.jpg',fin(plate(n),G))
# 6: slow fall with the dotted scale
n=240; t=n/25; G=np.zeros((H,W,3),np.float32); lively(G,t,amp=0.8); ya=apple_y(t)
for gy in ghosts:
    if gy<ya-20: add(G,GHOST,x0,gy,68,0.85)
add(G,APPLE,x0,ya,74); add(G,KNOB,1240,930,190,0.6,rot=80); add(G,SW,1690,300,400,0.9); scale(G,ya)
save('p6.jpg',fin(plate_lane(n),G))
# 7: time stop
n=350; G=np.zeros((H,W,3),np.float32); lively(G,13.7,frozen=True,amp=0.55)
for gy in ghosts: add(G,GHOST,x0,gy,68,0.7)
add(G,APPLE,x0,yF,76); add(G,SW,1690,300,400,0.85); add(G,KNOB,1240,930,190,0.5,rot=120); scale(G,yF,0.9)
ripple(G,x0,yF,180,(1.0,0.95,0.85),1.0); ripple(G,x0,yF,520,(1.0,0.95,0.85),0.5)
save('p7.jpg',fin(plate_lane(n,sat=0.25,tree_amp=0.75)*0.8+np.array([0.04,0.03,0.02]),G))
# 8: velocity question
n=445; G=np.zeros((H,W,3),np.float32); lively(G,13.7,frozen=True,amp=0.45)
for gy in ghosts: add(G,GHOST,x0,gy,68,0.55)
add(G,APPLE,x0,yF,76); add(G,SW,1690,300,400,0.85); scale(G,yF,0.8); ripple(G,x0,yF,62,(1.0,0.85,0.4),1.2)
Cc=np.zeros((H,W,3),np.float32); cv2.arrowedLine(Cc,(int(x0),int(yF+70)),(int(x0),int(yF+230)),(0.45,0.75,1.0),6,cv2.LINE_AA,tipLength=0.25)
text(Cc,'v = ?',(int(x0+40),int(yF+175)),1.6,(0.6,0.9,1.0),3)
G+=Cc+cv2.GaussianBlur(Cc,(0,0),6)*0.8
save('p8.jpg',fin(plate_lane(n,sat=0.25,tree_amp=0.75)*0.85,G))
print('ok')
