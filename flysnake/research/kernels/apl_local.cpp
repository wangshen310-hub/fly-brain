// Independent local APL candidate. Original kernel and old archives are untouched.
// Geometry queries are exact segment projections, not nearest soma/node guesses.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

struct BVHNode {std::array<double,3> lo,hi; int begin,end,left=-1,right=-1;};
struct SegmentTree {
 const double* xyz; const int32_t* child; const int32_t* parent;
 std::vector<int> order; std::vector<BVHNode> nodes;
 SegmentTree(int m,const double* x,const int32_t* c,const int32_t* p):xyz(x),child(c),parent(p),order(m){
   std::iota(order.begin(),order.end(),0);nodes.reserve(m*2);make(0,m);
 }
 int make(int begin,int end){
   BVHNode node;node.begin=begin;node.end=end;
   node.lo.fill(std::numeric_limits<double>::infinity());node.hi.fill(-std::numeric_limits<double>::infinity());
   for(int k=begin;k<end;k++)for(int d=0;d<3;d++){
     double a=xyz[3*child[order[k]]+d],b=xyz[3*parent[order[k]]+d];
     node.lo[d]=std::min(node.lo[d],std::min(a,b));node.hi[d]=std::max(node.hi[d],std::max(a,b));
   }
   int index=(int)nodes.size();nodes.push_back(node);
   if(end-begin>12){
     int axis=0;for(int d=1;d<3;d++)if(node.hi[d]-node.lo[d]>node.hi[axis]-node.lo[axis])axis=d;
     int middle=(begin+end)/2;
     std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](int a,int b){
       double aa=xyz[3*child[a]+axis]+xyz[3*parent[a]+axis];
       double bb=xyz[3*child[b]+axis]+xyz[3*parent[b]+axis];return aa==bb?a<b:aa<bb;
     });
     int left=make(begin,middle),right=make(middle,end);nodes[index].left=left;nodes[index].right=right;
   }
   return index;
 }
 double box_distance(int index,const double* q){
   double sum=0;for(int d=0;d<3;d++){
     double delta=std::max(nodes[index].lo[d]-q[d],std::max(0.,q[d]-nodes[index].hi[d]));sum+=delta*delta;
   }return sum;
 }
 void search(int index,const double* q,double& best,int& edge,double& fraction){
   if(box_distance(index,q)>best)return;
   const auto& node=nodes[index];
   if(node.left<0){
     for(int k=node.begin;k<node.end;k++){
       int e=order[k];double numerator=0,denominator=0;
       for(int d=0;d<3;d++){double delta=xyz[3*child[e]+d]-xyz[3*parent[e]+d];
         numerator+=(q[d]-xyz[3*parent[e]+d])*delta;denominator+=delta*delta;}
       double t=denominator>0?std::clamp(numerator/denominator,0.,1.):0.;double sq=0;
       for(int d=0;d<3;d++){double p=xyz[3*parent[e]+d]+t*(xyz[3*child[e]+d]-xyz[3*parent[e]+d]);sq+=(q[d]-p)*(q[d]-p);}
       if(sq<best || (sq==best && (edge<0 || e<edge))){best=sq;edge=e;fraction=t;}
     }
   }else{
     double a=box_distance(node.left,q),b=box_distance(node.right,q);
     if(a<=b){search(node.left,q,best,edge,fraction);search(node.right,q,best,edge,fraction);}
     else{search(node.right,q,best,edge,fraction);search(node.left,q,best,edge,fraction);}
   }
 }
};

extern "C" EXPORT void apl_attach_segments(int edges,const double* xyz,const int32_t* child,
 const int32_t* parent,int queries,const double* q,int32_t* result,double* fraction,double* distance){
 SegmentTree tree(edges,xyz,child,parent);
 for(int i=0;i<queries;i++){
   double best=std::numeric_limits<double>::infinity(),alpha=0;int edge=-1;
   tree.search(0,q+3*i,best,edge,alpha);result[i]=edge;fraction[i]=alpha;distance[i]=std::sqrt(best);
 }
}

// Finite-volume passive cable: volume*tau du/dt = -volume*u + axial +
// gE*(EE-rest-u) + gI*(EI-rest-u) + volume*local_drive.
// Constant-radius phenomenology. Published coarse SWC radii are NOT electrical fits.
// Hines elimination solves every disconnected tree separately; no cross-root wire.
static void cable_step(int n,double dt,const int32_t* parent,const double* axial,
 const double* volume,const double* diagonal_axial,double tau,double ee,double ei,
 double* u,double* ge,double* gi,const double* local_drive,double* diagonal,double* rhs){
 const double half=std::exp(-dt/10.),decay=half*half;
 for(int i=0;i<n;i++){
   double cap=volume[i]*tau/dt,e=ge[i]*half,in=gi[i]*half;
   diagonal[i]=cap+volume[i]+diagonal_axial[i]+e+in;
   rhs[i]=cap*u[i]+volume[i]*local_drive[i]+e*ee+in*ei;
 }
 for(int i=n-1;i>=0;i--)if(parent[i]>=0){
   int p=parent[i];double scale=axial[i]/diagonal[i];
   diagonal[p]-=axial[i]*scale;rhs[p]+=scale*rhs[i];
 }
 for(int i=0;i<n;i++){
   u[i]=(rhs[i]+(parent[i]>=0?axial[i]*u[parent[i]]:0.))/diagonal[i];
   ge[i]*=decay;gi[i]*=decay;
 }
}

extern "C" EXPORT void apl_cable_trace(int n,int steps,double dt,const int32_t* parent,
 const double* axial,const double* volume,const double* diagonal_axial,double tau,double ee,double ei,
 double* u,double* ge,double* gi,const double* drive,double* output){
 std::vector<double> diagonal(n),rhs(n);
 for(int t=0;t<steps;t++){
   cable_step(n,dt,parent,axial,volume,diagonal_axial,tau,ee,ei,u,ge,gi,drive,diagonal.data(),rhs.data());
   for(int i=0;i<n;i++)output[t*n+i]=u[i];
 }
}

// Same ABI as the archived event kernel, plus anatomically mapped local APL fields.
// All non-APL edges retain their original event semantics. APL output is graded.
extern "C" EXPORT void memory_advance_apl_local(
 int n,const int64_t* ptr,const int32_t* post,float* weight,
 float* v,float* g,int16_t* refractory,const float* drive,float* previous_drive,
 int32_t* queue,int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,int64_t* eligibility_last,
 int nplastic,const int64_t* plastic_edge,const int32_t* plastic_pre,
 const float* baseline_weight,const float* dan_gain,float eta,float tau_elig_ms,float floor_fraction,
 int learning_enabled,float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,const float* rest,
 float* adaptation,float adaptation_jump,float adaptation_tau,
 const uint8_t* apl_mask,int local_n,const int32_t* local_parent,const double* local_axial,
 const double* local_volume,const double* local_diagonal_axial,double* local_u,double* local_ge,double* local_gi,
 const double* local_drive,const int64_t* in_ptr,const int64_t* in_edge,const int32_t* in_child,
 const int32_t* in_parent,const double* in_alpha,const double* in_fraction,
 int out_n,const int64_t* out_edge,const int32_t* out_child,const int32_t* out_parent,
 const double* out_alpha,const double* out_fraction,const int32_t* out_target,
 const int32_t* out_local_child,const int32_t* out_local_parent,const double* out_local_alpha,
 int targets_n,const int32_t* targets,float* apl_inhibition,float* release_queue,double* local_release_queue,
 int local_stride,double tau,double input_reference_um,double release_half_mV,double release_max_hz,
 double release_gain,float inhibitory_reversal){
 const int delay=std::lround(1.8f/dt),rfc=std::lround(2.2f/dt),slots=delay+1;
 float av[1024],ag[1024],aa[1024];
 for(int i=0;i<1024;i++){av[i]=std::exp(-dt*i/20.f);ag[i]=std::exp(-dt*i/5.f);aa[i]=std::exp(-dt*i/adaptation_tau);}
 std::vector<double> diagonal(local_n),rhs(local_n);
 auto evolve=[&](int i,int64_t now,float current){
   int64_t d=now-last[i];if(d<=0)return;
   if(apl_inhibition[i]==0.f){
     const int frozen=refractory[i]>0?refractory[i]-1:0;const int skip=(int)(d<frozen?d:frozen);
     if(skip>0 && adaptation[i]>0)adaptation[i]*=skip<1024?aa[skip]:std::exp(-dt*skip/adaptation_tau);
     refractory[i]=d>=refractory[i]?0:refractory[i]-d;d-=skip;
     if(d>0){const float a=d<1024?av[d]:std::exp(-dt*d/20.f),b=d<1024?ag[d]:std::exp(-dt*d/5.f);
       v[i]=rest[i]+(v[i]-rest[i])*a+current*(1.f-a)+g[i]*(a-b)/3.f;g[i]*=b;
       if(adaptation[i]>0){const float c=d<1024?aa[d]:std::exp(-dt*d/adaptation_tau);
         v[i]-=adaptation[i]*adaptation_tau/(adaptation_tau-20.f)*(c-a);adaptation[i]*=c;}
     }
   }else{
     // Only APL's new receptor is conductance-based. Other old current sources
     // can still violate physiological bounds; full-graph assays must check them.
     for(int64_t k=0;k<d;k++){
       if(refractory[i]>1){refractory[i]--;adaptation[i]*=aa[1];}
       else{
         refractory[i]=0;float in=apl_inhibition[i]*std::sqrt(ag[1]);
         float a=adaptation[i]*std::sqrt(aa[1]),syn=g[i]*std::sqrt(ag[1]);
         float coefficient=1.f+in;
         float target=(rest[i]+current+syn-a+in*inhibitory_reversal)/coefficient;
         v[i]=target+(v[i]-target)*std::exp(-coefficient*dt/20.f);
         g[i]*=ag[1];adaptation[i]*=aa[1];
       }
       apl_inhibition[i]*=ag[1];
     }
   }
   last[i]=now;
 };
 auto awaken=[&](int i){if(!flags[i]){flags[i]=1;active[(*nactive)++]=i;}};
 for(int i=0;i<n;i++)if(drive[i]!=previous_drive[i]){
   evolve(i,*clock-1,previous_drive[i]);previous_drive[i]=drive[i];if(!apl_mask[i])awaken(i);
 }
 for(int t=0;t<steps;t++,(*clock)++){
   int slot=*clock%slots,future=(*clock+delay)%slots;
   int kept=0,original=*nactive;
   for(int k=0;k<original;k++){
     int i=active[k];if(apl_mask[i]){flags[i]=0;continue;}evolve(i,*clock,drive[i]);
     if(refractory[i]==0 && v[i]>-45.f){queue[future*n+queue_count[future]++]=i;counts[i]++;
       if(kc_mask[i]){adaptation[i]+=adaptation_jump;
         eligibility[i]*=std::exp(-dt*(*clock-eligibility_last[i])/tau_elig_ms);
         eligibility[i]+=1.;eligibility_last[i]=*clock;}
     }
     float gap=-45.f-rest[i];
     bool can_fire=v[i]>-45.f || drive[i]>gap || drive[i]+g[i]>gap;
     if(can_fire)active[kept++]=i;else flags[i]=0;
   }
   *nactive=kept;
   for(int q=0;q<queue_count[slot];q++){
     int i=queue[slot*n+q];
     if(modulation_mask[i]){
       for(int64_t e=ptr[i];e<ptr[i+1];e++){
         int j=post[e];modulation[j]*=std::exp(-dt*(*clock-modulation_last[j])/100.f);
         modulation[j]+=std::abs(weight[e])/.275f;modulation_last[j]=*clock;
       }
       if(learning_enabled && dan_index[i]>=0)for(int p=0;p<nplastic;p++){
         int pre=plastic_pre[p];double trace=eligibility[pre]*std::exp(-dt*(*clock-eligibility_last[pre])/tau_elig_ms);
         float gain=dan_gain[dan_index[i]*nplastic+p];int64_t edge=plastic_edge[p];
         float candidate=weight[edge]*std::exp(-eta*gain*trace),lower=baseline_weight[p]*floor_fraction;
         weight[edge]=candidate>lower?candidate:lower;
       }
       continue;
     }
     for(int64_t e=ptr[i];e<ptr[i+1];e++){
       int j=post[e];if(apl_mask[j])continue;evolve(j,*clock,drive[j]);
       if(refractory[j]==0){g[j]+=weight[e];awaken(j);}
     }
     for(int64_t c=in_ptr[i];c<in_ptr[i+1];c++){
       double w=weight[in_edge[c]]*in_fraction[c]*input_reference_um;
       double* state=w>=0?local_ge:local_gi;double increment=w>=0?w/52.:w/(inhibitory_reversal+52.);
       state[in_child[c]]+=increment*in_alpha[c];state[in_parent[c]]+=increment*(1.-in_alpha[c]);
     }
   }
   queue_count[slot]=0;
   // Graded releases retain the same 1.8-ms axonal delivery delay. The new
   // receptor accepts release while a target is refractory and keeps decaying.
   for(int k=0;k<targets_n;k++){
     int j=targets[k];float& queued=release_queue[slot*targets_n+k];
     if(queued!=0.f){evolve(j,*clock,drive[j]);apl_inhibition[j]+=queued;awaken(j);queued=0.f;}
   }
   for(int j=0;j<local_n;j++){
     local_gi[j]+=local_release_queue[slot*local_n+j];local_release_queue[slot*local_n+j]=0.;
   }
   for(int q=0;q<queue_count[future];q++){
     int i=queue[future*n+q];v[i]=rest[i];g[i]=0.f;refractory[i]=rfc;
   }
   if(local_n && (*clock+1)%local_stride==0){
     double interval=dt*local_stride;
     cable_step(local_n,interval,local_parent,local_axial,local_volume,local_diagonal_axial,
       tau,52.,inhibitory_reversal+52.,local_u,local_ge,local_gi,local_drive,diagonal.data(),rhs.data());
     for(int c=0;c<out_n;c++){
       double u=out_alpha[c]*local_u[out_child[c]]+(1.-out_alpha[c])*local_u[out_parent[c]];
       double positive=std::max(0.,u),rate=release_max_hz*positive/(release_half_mV+positive);
       double amount=-weight[out_edge[c]]*out_fraction[c]*rate*interval/1000.*release_gain;
       if(out_target[c]>=0){int k=out_target[c],j=targets[k];
         release_queue[future*targets_n+k]+=(float)(amount/(rest[j]-inhibitory_reversal));
       }else{
         double increment=amount/( -52.-inhibitory_reversal)*input_reference_um;
         local_release_queue[future*local_n+out_local_child[c]]+=increment*out_local_alpha[c];
         local_release_queue[future*local_n+out_local_parent[c]]+=increment*(1.-out_local_alpha[c]);
       }
     }
   }
 }
 for(int i=0;i<n;i++)evolve(i,*clock-1,drive[i]);
}
