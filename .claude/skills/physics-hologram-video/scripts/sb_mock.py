import cv2, numpy as np, math
A='assets'; W,H=1920,1080
P = np.load('palms.npy')
yy, xx = np.mgrid[0:H,0:W].astype(np.float32)
NAVY = np.array([0.05,0.02,0.006],np.float32)*np.exp(-((xx-900)**2+(yy-560)**2)/(2*700.0**2))[...,None]
KEY = 0.3+0.85*np.exp(-((xx-1050)**2+((yy-470)*1.05)**2)/(2*330.0**2))
T = cv2.imread(A+'/tree2.png', cv2.IMREAD_UNCHANGED).astype(np.float32)/255
TR = cv2.warpAffine(T, np.array([[0.8,0,13],[0,0.8,67]],np.float32), (W,H))
APPLE=cv2.imread(A+'/apple.png').astype(np.float32)/255; KNOB=cv2.imread(A+'/knob.png').astype(np.float32)/255; SW=cv2.imread(A+'/stopwatch.png').astype(np.float32)/255
def text(C, s, org, scale, col, th):
    m=np.zeros((H,W),np.uint8); cv2.putText(m,s,org,cv2.FONT_HERSHEY_SIMPLEX,scale,255,th,cv2.LINE_AA)
    C+= (m.astype(np.float32)/255)[...,None]*np.array(col,np.float32)
def add(G,img,cx,cy,h,amp=1.0,rot=0):
    s=h/img.shape[0]; im=cv2.resize(img,(int(img.shape[1]*s),int(h)),interpolation=cv2.INTER_AREA)
    if rot: im=cv2.warpAffine(im, cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2),rot,1), (im.shape[1],im.shape[0]))
    x0,y0=int(cx-im.shape[1]/2),int(cy-im.shape[0]/2); sx,sy=max(0,-x0),max(0,-y0); dx,dy=max(0,x0),max(0,y0)
    ex,ey=min(W,x0+im.shape[1]),min(H,y0+im.shape[0]); G[dy:ey,dx:ex]+=im[sy:sy+ey-dy,sx:sx+ex-dx]*amp
def plate(n, sat=1.0, tree_amp=0.93):
    a=cv2.imread('masks_rvm/%05d.png'%(n+1),0).astype(np.float32)/255; fg=cv2.imread('fgr_rvm/%05d.png'%(n+1)).astype(np.float32)/255
    PG=(1-np.exp(-2*fg*0.95))/(1-math.exp(-2))*KEY[...,None]*np.array([1.06,0.98,0.92],np.float32)
    rgb=TR[...,:3]
    if sat<1: g=rgb.mean(2,keepdims=True); rgb=rgb*sat+g*(1-sat)*np.array([1.1,1.0,0.85])
    ta=TR[...,3:4]*tree_amp
    return PG*a[...,None]+(NAVY*(1-ta)+rgb*ta)*(1-a[...,None])
def fin(base,G): return np.clip(1-(1-base)*(1-np.clip(G,0,1)),0,1)
x0, y0 = P[int(4.22*25),0][0]+4, P[int(4.22*25),0][1]-16
yF = 775.0
def apple_y(t):
    y76 = P[int(7.6*25),0][1]-16
    if t < 7.6: return y0 + (y76-y0)*((t-4.22)/(7.6-4.22))**1.6
    return y76 + (yF-y76)*min(1,(1-math.exp(-(min(t,13.7)-7.6)/2.2))/(1-math.exp(-6.1/2.2)))
ghosts=[y0+(yF-y0)*(k/6)**2 for k in range(1,6)]
def ripple(G,cx,cy,r,col,amp):
    C=np.zeros((H,W),np.float32); cv2.circle(C,(int(cx),int(cy)),int(r),1.0,3,cv2.LINE_AA)
    G+=(cv2.GaussianBlur(C,(0,0),4)*amp)[...,None]*np.array(col,np.float32)
n=70; G=np.zeros((H,W,3),np.float32); hp=P[n,0]; fp=P[n,1]
add(G,APPLE,hp[0]+4,hp[1]-16,72); add(G,KNOB,fp[0]+5,fp[1]-55,190,1.0,rot=40); add(G,SW,1690,300,400)
ripple(G,fp[0],fp[1]-55,260,(1.0,0.75,0.3),0.9); ripple(G,fp[0],fp[1]-55,420,(1.0,0.75,0.3),0.45)
cv2.imwrite('sb/mock_2.8.jpg',(fin(plate(n),G)*255).astype(np.uint8))
n=240; t=n/25; G=np.zeros((H,W,3),np.float32)
for gy in ghosts:
    if gy < apple_y(t)-20: add(G,APPLE,x0,gy,70,0.7)
add(G,APPLE,x0,apple_y(t),72); add(G,KNOB,1240,930,190,0.6,rot=80); add(G,SW,1690,300,400,0.9)
pts=[y0]+[g for g in ghosts if g<apple_y(t)-20]+[apple_y(t)]
for i,(a_,b_) in enumerate(zip(pts[:-1],pts[1:])):
    C=np.zeros((H,W,3),np.float32); X=int(x0-75)
    cv2.line(C,(X,int(a_)),(X,int(b_)),(1.0,0.85,0.4),2,cv2.LINE_AA); cv2.line(C,(X-10,int(b_)),(X+10,int(b_)),(1.0,0.85,0.4),2,cv2.LINE_AA)
    text(C,'d%d'%(i+1),(X-62,int((a_+b_)/2)+8),0.8,(1.0,0.9,0.6),2)
    G+=C+cv2.GaussianBlur(C,(0,0),4)*0.6
cv2.imwrite('sb/mock_9.6.jpg',(fin(plate(n),G)*255).astype(np.uint8))
n=350; G=np.zeros((H,W,3),np.float32)
for gy in ghosts: add(G,APPLE,x0,gy,70,0.3)
add(G,APPLE,x0,yF,74); add(G,SW,1690,300,400,0.85); add(G,KNOB,1240,930,190,0.5,rot=120)
ripple(G,x0,yF,180,(1.0,0.95,0.85),1.0); ripple(G,x0,yF,520,(1.0,0.95,0.85),0.5)
base=plate(n,sat=0.25,tree_amp=0.75)*0.8+np.array([0.04,0.03,0.02])
cv2.imwrite('sb/mock_14.0.jpg',(fin(base,G)*255).astype(np.uint8))
n=445; G=np.zeros((H,W,3),np.float32)
for gy in ghosts: add(G,APPLE,x0,gy,70,0.25)
add(G,APPLE,x0,yF,74); add(G,SW,1690,300,400,0.85); ripple(G,x0,yF,62,(1.0,0.85,0.4),1.2)
C=np.zeros((H,W,3),np.float32); cv2.arrowedLine(C,(int(x0),int(yF+70)),(int(x0),int(yF+230)),(0.45,0.75,1.0),6,cv2.LINE_AA,tipLength=0.25)
text(C,'v = ?',(int(x0+40),int(yF+175)),1.6,(0.6,0.9,1.0),3)
G+=C+cv2.GaussianBlur(C,(0,0),6)*0.8
cv2.imwrite('sb/mock_17.8.jpg',(fin(plate(n,sat=0.25,tree_amp=0.75)*0.85,G)*255).astype(np.uint8))
print('ok', round(x0), round(y0), [round(g) for g in ghosts])
