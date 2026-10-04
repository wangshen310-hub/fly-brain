// Independent full-connectome frozen diagnostic. Never replaces kernel.cpp.
// Signed existing weights are converted to positive E/I conductances using
// the target's resting driving force. APL is STILL a spiking point neuron;
// this variant does not claim to reconstruct graded/local APL physiology.
#include <cmath>
#include <cstdint>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

struct Evolution {
 float dt, ee, ei, syn_decay, syn_half, adapt_decay, adapt_half;
 Evolution(float step,float ex,float in,float adapt_tau):dt(step),ee(ex),ei(in),
   syn_decay(std::exp(-step/5.f)),syn_half(std::exp(-step/10.f)),
   adapt_decay(std::exp(-step/adapt_tau)),adapt_half(std::exp(-step/(2.f*adapt_tau))) {}
 void tick(float& v,float& ge,float& gi,float& a,int16_t& ref,float rest,float drive) const {
   const float e=ge*syn_half,i=gi*syn_half,ad=a*adapt_half;
   // Frozen-coefficient exponential midpoint; second order in dt, not the
   // exact continuous solution for time-varying synaptic conductances.
   if(ref>0){v=rest;--ref;}
   else if(ge!=0.f || gi!=0.f || a!=0.f || drive!=0.f || v!=rest){
     const float k=1.f+e+i;
     const float target=(rest+drive-ad+e*ee+i*ei)/k;
     v=target+(v-target)*std::exp(-k*dt/20.f);
   }
   ge*=syn_decay;gi*=syn_decay;a*=adapt_decay;
 }
};

// Small numerical validation export. It is not a behavioral controller or
// substituted neural response; tests compare it with an independent ODE.
extern "C" EXPORT void conductance_cell_trace(int steps,float dt,float ee,float ei,
 float rest,float drive,float adapt_tau,float* state,int16_t* ref,float* output){
 Evolution evolution(dt,ee,ei,adapt_tau);
 for(int t=0;t<steps;t++){
   evolution.tick(state[0],state[1],state[2],state[3],*ref,rest,drive);
   for(int j=0;j<4;j++)output[t*4+j]=state[j];
 }
}

// Existing MemoryBrain ABI plus two independent conductance states and E/I
// reversal potentials. Learning is intentionally rejected by the Python bridge.
extern "C" EXPORT void memory_advance_conductance(
 int n,const int64_t* ptr,const int32_t* post,float* weight,
 float* v,float* legacy_g,int16_t* refractory,const float* drive,float* previous_drive,
 int32_t* queue,int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,int64_t* eligibility_last,
 int nplastic,const int64_t* plastic_edge,const int32_t* plastic_pre,
 const float* baseline_weight,const float* dan_gain,float eta,float tau_elig_ms,float floor_fraction,
 int learning_enabled,float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,const float* rest,
 float* adaptation,float adaptation_jump,float adaptation_tau,
 float* ge,float* gi,float ee,float ei){
 const int delay=std::lround(1.8f/dt),rfc=std::lround(2.2f/dt),slots=delay+1;
 Evolution evolution(dt,ee,ei,adaptation_tau);
 for(int t=0;t<steps;t++,(*clock)++){
   const int slot=*clock%slots,future=(*clock+delay)%slots;
   // Every retained neuron is considered at every original 0.1-ms tick.
   // No dropped edges, activity threshold, reduced graph or policy readout.
   for(int i=0;i<n;i++){
     const bool was_refractory=refractory[i]>0;
     evolution.tick(v[i],ge[i],gi[i],adaptation[i],refractory[i],rest[i],drive[i]);
     if(!was_refractory && v[i]>-45.f){
       queue[future*n+queue_count[future]++]=i;counts[i]++;
       if(kc_mask[i]){
         adaptation[i]+=adaptation_jump;
         eligibility[i]*=std::exp(-dt*(*clock-eligibility_last[i])/tau_elig_ms);
         eligibility[i]+=1.;eligibility_last[i]=*clock;
       }
     }
   }
   for(int q=0;q<queue_count[slot];q++){
     const int i=queue[slot*n+q];
     if(modulation_mask[i]){
       for(int64_t edge=ptr[i];edge<ptr[i+1];edge++){
         const int j=post[edge];
         modulation[j]*=std::exp(-dt*(*clock-modulation_last[j])/100.f);
         modulation[j]+=std::abs(weight[edge])/.275f;modulation_last[j]=*clock;
       }
       // No learning in this diagnostic: no KC-to-MBON weights are written.
       continue;
     }
     for(int64_t edge=ptr[i];edge<ptr[i+1];edge++){
       const int j=post[edge];const float w=weight[edge];
       // Unlike the retained current model, synaptic receptor state persists
       // across spikes/refractory intervals; incoming events remain accepted.
       if(w>=0.f)ge[j]+=w/(ee-rest[j]);
       else gi[j]+=w/(ei-rest[j]);
     }
   }
   queue_count[slot]=0;
   for(int q=0;q<queue_count[future];q++){
     const int i=queue[future*n+q];v[i]=rest[i];refractory[i]=rfc;
   }
 }
 for(int i=0;i<n;i++){
   legacy_g[i]=ge[i]*(ee-v[i])+gi[i]*(ei-v[i]);
   previous_drive[i]=drive[i];last[i]=*clock-1;
 }
 // Legacy lazy-active fields no longer select dynamics; clear for auditability.
 *nactive=0;for(int i=0;i<n;i++)flags[i]=0;
}
