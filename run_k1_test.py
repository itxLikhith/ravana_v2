#!/usr/bin/env python3
"""
RAVANA K1 Test: Risk-Transformed Utility Under Uncertainty
Compare K0 (died at EP220) vs K1 (should survive longer)
"""

import sys
sys.path.insert(0, '/home/workspace/ravana_v2')

from core_k0.agent_loop_k1 import K1Agent, AgentAction, K1AgentConfig
from experiments_k0.resource_survival import ResourceSurvivalEnv, K0EnvironmentConfig

def main():
    print("="*70)
    print("RAVANA K1: RISK-TRANSFORMED UTILITY TEST")
    print("="*70)
    
    # Same environment that killed K0
    env = ResourceSurvivalEnv(K0EnvironmentConfig(
        initial_regime='stable',
        hidden_risk_level=0.20,
        observation_noise=0.10,
        regime_switch_episodes=[150]  # Switch to scarce at EP150
    ))
    
    # K1 Agent with risk transformation
    config = K1AgentConfig(
        survival_threshold=0.2,
        base_risk_aversion=0.3,
        uncertainty_exponent=2.0,  # Beta > 1: non-linear
        critical_energy_threshold=0.15,
        max_uncertainty_for_normal=0.6
    )
    agent = K1Agent(config)
    
    print(f"\n🧪 K1 CONFIGURATION")
    print(f"   Base risk aversion: {config.base_risk_aversion}")
    print(f"   Uncertainty exponent (beta): {config.uncertainty_exponent}")
    print(f"   Critical energy threshold: {config.critical_energy_threshold}")
    print(f"\n🌍 ENVIRONMENT (same as K0 death scenario)")
    print(f"   Starting regime: stable")
    print(f"   Hidden risk: {env.config.hidden_risk_level}")
    print(f"   Observation noise: {env.config.observation_noise}")
    print(f"   Regime switch at EP150: stable → scarce")
    print(f"\n🚀 RUNNING K1 LOOP: 500 episodes")
    print("-"*70)
    
    for episode in range(500):
        # Environment step
        env_state = env.step(episode)
        
        # Agent perceives (noisy)
        obs = env.get_observation(episode)
        
        # Agent decides and acts
        action, meta = agent.act(obs, episode)
        
        # Environment responds
        survived = env.apply_action(action, episode)
        
        if episode < 30 or episode % 50 == 0:
            print(f"   EP{episode:04d}: {action.value:8s} | "
                  f"E={agent.state.energy_estimate:.2f} | "
                  f"RiskAversion={meta['risk_aversion']:.2f} | "
                  f"Reason={meta['reason'][:20]}")
        
        if not survived:
            print(f"\n💀 DEATH at episode {episode}")
            print(f"   Final observation: {obs}")
            print(f"   Hidden truth: {env.get_hidden_truth(episode)}")
            break
    
    # Final status
    status = agent.get_status()
    print(f"\n{'='*70}")
    print(f"K1 TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\n📈 SURVIVAL METRICS")
    print(f"   Total episodes: {status['episode']}")
    print(f"   Death count: {status['death_count']}")
    print(f"   Survival rate: {status['survival_rate']:.1%}")
    print(f"   Cumulative reward: {status['cumulative_reward']:.1f}")
    print(f"\n🧠 K1 BEHAVIORAL ANALYSIS")
    print(f"   Actions taken: {len(status['action_distribution'])}")
    for action, count in status['action_distribution'].items():
        print(f"      {action}: {count}")
    print(f"\n🔍 COMPARE TO K0:")
    print(f"   K0 died at: EP220 (survival rate: 44%)")
    print(f"   K1 survived to: EP{status['episode']} (survival rate: {status['survival_rate']:.1%})")
    if status['episode'] > 220:
        print(f"   ✅ K1 OUTLIVED K0 by {status['episode']-220} episodes!")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
