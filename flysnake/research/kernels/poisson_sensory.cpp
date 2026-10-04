// Experimental dopamine-gated KC-to-MBON11 depression.
// Baseline integration is copied without changing its timestep/schedule.
// PPL101 edges deliver a modeled modulatory trace instead of fast excitation.
// Lazy exact subthreshold evolution. An inactive neuron is skipped only when
// its voltage AND both its instantaneous and asymptotic drive are below the
// threshold. With no incoming event it cannot fire. Every edge is retained.
#include <cmath>
#include <cstdint>
#include <vector>
#if defined(_WIN32)
#define FLYSNAKE_EXPORT __declspec(dllexport)
#else
#define FLYSNAKE_EXPORT
#endif
// Independent Poisson sensory-event extension. No-input behavior is retained.
// Event stage matches a declared synapses-stage voltage jump. Only externally
// driven receptor targets are assigned zero refractory, as in Shiu model.py.
static thread_local const int64_t* input_ticks=nullptr;
static thread_local const int32_t* input_targets=nullptr;
static thread_local const uint8_t* input_mask=nullptr;
static thread_local int input_count=0,input_cursor=0;
static thread_local float input_jump=0.f;
extern "C" FLYSNAKE_EXPORT void memory_advance(
 int n,const int64_t* ptr,const int32_t* post,float* weight,
 float* v,float* g,int16_t* refractory,const float* drive,float* previous_drive,
 int32_t* queue,int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,int64_t* eligibility_last,
 int nplastic,const int64_t* plastic_edge,const int32_t* plastic_pre,
 const float* baseline_weight,const float* dan_gain,float eta,float tau_elig_ms,float floor_fraction,
 int learning_enabled,float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,const float* rest,
 float* adaptation,float adaptation_jump,float adaptation_tau) {
 const int delay=std::lround(1.8f/dt),rfc=std::lround(2.2f/dt),slots=delay+1;
 float av[1024],ag[1024],aa[1024];
 for(int i=0;i<1024;i++){av[i]=std::exp(-dt*i/20.f);ag[i]=std::exp(-dt*i/5.f);aa[i]=std::exp(-dt*i/adaptation_tau);}
 auto evolve=[&](int i,int64_t now,float current){
   int64_t d=now-last[i];if(d<=0)return;
   const int frozen=refractory[i]>0?refractory[i]-1:0;
   const int skip=(int)(d<frozen?d:frozen);
   if(skip>0 && adaptation[i]>0)adaptation[i]*=skip<1024?aa[skip]:std::exp(-dt*skip/adaptation_tau);
   refractory[i]=d>=refractory[i]?0:refractory[i]-d;d-=skip;
   if(d>0){const float a=d<1024?av[d]:std::exp(-dt*d/20.f),b=d<1024?ag[d]:std::exp(-dt*d/5.f);
     v[i]=rest[i]+(v[i]-rest[i])*a+current*(1.f-a)+g[i]*(a-b)/3.f;g[i]*=b;
     if(adaptation[i]>0){const float c=d<1024?aa[d]:std::exp(-dt*d/adaptation_tau);
       v[i]-=adaptation[i]*adaptation_tau/(adaptation_tau-20.f)*(c-a);adaptation[i]*=c;}
   }
   last[i]=now;
 };
 auto awaken=[&](int i){if(!flags[i]){flags[i]=1;active[(*nactive)++]=i;}};
 // Apply changing sensory currents only after settling old-current history.
 for(int i=0;i<n;i++)if(drive[i]!=previous_drive[i]){
   evolve(i,*clock-1,previous_drive[i]);previous_drive[i]=drive[i];awaken(i);
 }
 for(int t=0;t<steps;t++,(*clock)++){
   const int slot=*clock%slots,future=(*clock+delay)%slots;
   int kept=0,original=*nactive;
   for(int k=0;k<original;k++){
     const int i=active[k];evolve(i,*clock,drive[i]);
     if(refractory[i]==0 && v[i]>-45.f){queue[future*n+queue_count[future]++]=i;counts[i]++;
       if(kc_mask[i]){
         adaptation[i]+=adaptation_jump;
         eligibility[i]*=std::exp(-dt*(*clock-eligibility_last[i])/tau_elig_ms);
         eligibility[i]+=1.;eligibility_last[i]=*clock;
       }
     }
     // Convex relaxation toward drive+g(t): the bound makes this exact in
     // real arithmetic, not an activity cutoff or a dropped weak connection.
     const float gap=-45.f-rest[i];
     const bool can_fire=v[i]>-45.f || drive[i]>gap || drive[i]+g[i]>gap;
     if(can_fire)active[kept++]=i;else flags[i]=0;
   }
   *nactive=kept;
   for(int q=0;q<queue_count[slot];q++){
     const int i=queue[slot*n+q];
     if(modulation_mask[i]){
       // Keep all outgoing reconstructed edges: deliver transmitter to each
       // actual target. Effects outside the specified memory rule are unknown.
       for(int64_t e=ptr[i];e<ptr[i+1];e++){
         const int j=post[e];
         modulation[j]*=std::exp(-dt*(*clock-modulation_last[j])/100.f);
         modulation[j]+=std::abs(weight[e])/.275f;modulation_last[j]=*clock;
       }
       if(learning_enabled && dan_index[i]>=0){
         for(int p=0;p<nplastic;p++){
           const int pre=plastic_pre[p];
           const double trace=eligibility[pre]*std::exp(-dt*(*clock-eligibility_last[pre])/tau_elig_ms);
           const float gain=dan_gain[dan_index[i]*nplastic+p];
           const int64_t edge=plastic_edge[p];
           const float candidate=weight[edge]*std::exp(-eta*gain*trace);
           const float lower=baseline_weight[p]*floor_fraction;
           weight[edge]=candidate>lower?candidate:lower;
         }
       }
       continue;
     }
     for(int64_t e=ptr[i];e<ptr[i+1];e++){
       const int j=post[e];evolve(j,*clock,drive[j]);
       if(refractory[j]==0){g[j]+=weight[e];awaken(j);}
     }
   }
   while(input_cursor<input_count && input_ticks[input_cursor]==*clock){
     const int j=input_targets[input_cursor++];
     evolve(j,*clock,drive[j]);v[j]+=input_jump;awaken(j);
   }
   queue_count[slot]=0;
   for(int q=0;q<queue_count[future];q++){
     const int i=queue[future*n+q];v[i]=rest[i];g[i]=0.f;refractory[i]=(input_mask && input_mask[i])?0:rfc;
   }
 }
 // Materialize all states at the observation boundary (no threshold can be
 // missed in sleeping cells). This also supports auditable voltage readouts.
 for(int i=0;i<n;i++)evolve(i,*clock-1,drive[i]);
}

extern "C" FLYSNAKE_EXPORT void memory_advance_inputs(
 int n,const int64_t* ptr,const int32_t* post,float* weight,
 float* v,float* g,int16_t* refractory,const float* drive,float* previous_drive,
 int32_t* queue,int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,int64_t* eligibility_last,
 int nplastic,const int64_t* plastic_edge,const int32_t* plastic_pre,
 const float* baseline_weight,const float* dan_gain,float eta,float tau_elig_ms,float floor_fraction,
 int learning_enabled,float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,const float* rest,
 float* adaptation,float adaptation_jump,float adaptation_tau,
 int nevents,const int64_t* event_ticks,const int32_t* event_targets,
 const uint8_t* external_input_mask,float voltage_jump) {
 input_ticks=event_ticks;input_targets=event_targets;input_mask=external_input_mask;
 input_count=nevents;input_cursor=0;input_jump=voltage_jump;
 memory_advance(n,ptr,post,weight,v,g,refractory,drive,previous_drive,queue,queue_count,clock,steps,dt,counts,active,flags,nactive,last,kc_mask,dan_index,eligibility,eligibility_last,nplastic,plastic_edge,plastic_pre,baseline_weight,dan_gain,eta,tau_elig_ms,floor_fraction,learning_enabled,modulation,modulation_last,modulation_mask,rest,adaptation,adaptation_jump,adaptation_tau);
 input_ticks=nullptr;input_targets=nullptr;input_mask=nullptr;
 input_count=0;input_cursor=0;input_jump=0.f;
}
