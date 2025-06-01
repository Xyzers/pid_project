# src/pid_logic/pid_controller.py
import logging

logger = logging.getLogger(__name__)

class PIDController:
    def __init__(self, Kp, Ti, Td, Tsamp, mv_min, mv_max,
                 direct_action=True, initial_mv=0.0,
                 pid_structure='parallel_isa',
                 derivative_action='on_pv'):
        
        self.Kp_param = Kp
        self.Ti_param = Ti if Ti > 0 else float('inf')
        self.Td_param = Td
        self.Tsamp = Tsamp 
        self.mv_min = mv_min
        self.mv_max = mv_max
        self.direct_action = direct_action
        
        if pid_structure not in ['parallel_isa', 'parallel_kp_global']:
            raise ValueError(f"pid_structure invalide : {pid_structure}. Choisir 'parallel_isa' ou 'parallel_kp_global'.")
        self.pid_structure = pid_structure

        if derivative_action not in ['on_pv', 'on_error']:
            raise ValueError(f"derivative_action invalide : {derivative_action}. Choisir 'on_pv' ou 'on_error'.")
        self.derivative_action = derivative_action

        # --- États internes du PID ---
        # self.integral_term est TOUJOURS le terme Intégral complet (ex: Ki * somme(erreurs*dt))
        # prêt à être sommé avec P et D (si structure ISA) ou multiplié par Kp (si Kp global et I_term est alors interne)
        # Pour simplifier, nous allons faire que self.integral_term est l'accumulateur que nous allons construire,
        # et son interprétation dépendra de ki_calc et de la structure.
        # Plus précisément:
        # Pour ISA: self.integral_term accumule (Kp*Ts/Ti)*erreur. C'est le terme I final.
        # Pour Kp_global: self.integral_term accumule (Ts/Ti)*erreur. C'est I_interne.
        self.integral_term = 0.0
        
        self.previous_pv = None
        self.previous_error = None
        self.mv = initial_mv 
        self.is_active = True 
        self.last_active_mv = initial_mv

        # Ces variables stockeront les composants calculés à chaque pas (pour logging/debug si besoin)
        self.p_component = 0.0
        self.i_component = 0.0 # Sera self.integral_term
        self.d_component = 0.0

        self._update_internal_gains()

    def _update_internal_gains(self):
        self.kp_actual = self.Kp_param # Gain P ou gain global

        if self.pid_structure == 'parallel_isa':
            # Kp est dans chaque terme. Ki = Kp*Ts/Ti, Kd = Kp*Td/Ts
            self.ki_actual = (self.Kp_param * self.Tsamp / self.Ti_param) if (self.Ti_param > 0 and self.Tsamp > 0) else 0.0
            self.kd_actual = (self.Kp_param * self.Td_param / self.Tsamp) if (self.Tsamp > 0 and self.Td_param > 0) else 0.0
        elif self.pid_structure == 'parallel_kp_global':
            # Kp est appliqué globalement. ki_interne = Ts/Ti, kd_interne = Td/Ts
            self.ki_actual = (self.Tsamp / self.Ti_param) if (self.Ti_param > 0 and self.Tsamp > 0) else 0.0
            self.kd_actual = (self.Td_param / self.Tsamp) if (self.Tsamp > 0 and self.Td_param > 0) else 0.0
        
        # logger.debug(f"Gains PID mis à jour: Kp_actual={self.kp_actual:.3f}, Ki_actual={self.ki_actual:.3f}, Kd_actual={self.kd_actual:.3f} (Structure: {self.pid_structure})")


    def set_parameters(self, Kp, Ti, Td):
        parameter_changed = False
        if self.Kp_param != Kp: self.Kp_param = Kp; parameter_changed = True
        new_ti_param = Ti if Ti > 0 else float('inf')
        if self.Ti_param != new_ti_param: self.Ti_param = new_ti_param; parameter_changed = True
        if self.Td_param != Td: self.Td_param = Td; parameter_changed = True
        if parameter_changed:
            self._update_internal_gains()

    def _limit_mv(self, mv_candidate):
        return max(self.mv_min, min(mv_candidate, self.mv_max))

    def set_initial_state(self, pv_initial, sp_initial, mv_initial, active_initial):
        self.previous_pv = pv_initial
        self.mv = self._limit_mv(mv_initial)
        self.last_active_mv = self.mv
        self.is_active = active_initial

        current_error = sp_initial - pv_initial
        if not self.direct_action: current_error = -current_error
        self.previous_error = current_error

        if self.is_active:
            self.integral_term = self._calculate_integral_for_bumpless(sp_initial, pv_initial, self.mv)
        else: 
            self.integral_term = self._calculate_integral_for_bumpless(sp_initial, pv_initial, self.last_active_mv)
        
        # logger.debug(f"État initial: PV={pv_initial:.2f}, SP={sp_initial:.2f}, MV={self.mv:.2f}, Active={self.is_active}, I_term={self.integral_term:.3f}")


    def _calculate_integral_for_bumpless(self, sp, pv, target_mv_for_bumpless):
        # Calcule la valeur que self.integral_term doit avoir pour que
        # la sortie PID (P+I+D ou Kp*(P_int+I_int+D_int)) égale target_mv_for_bumpless
        if self.previous_pv is None: self.previous_pv = pv 

        current_error = sp - pv
        if not self.direct_action: current_error = -current_error
        if self.previous_error is None: self.previous_error = current_error

        p_val = 0.0
        d_val = 0.0

        # Calcul P et D comme dans update()
        if self.pid_structure == 'parallel_isa':
            p_val = self.kp_actual * current_error
            if self.derivative_action == 'on_pv' and self.kd_actual != 0 and self.previous_pv is not None:
                delta_pv = pv - self.previous_pv
                d_val = -self.kd_actual * delta_pv
            elif self.derivative_action == 'on_error' and self.kd_actual != 0 and self.previous_error is not None:
                delta_error = current_error - self.previous_error
                d_val = self.kd_actual * delta_error
            
            # target_mv = p_val + integral_term_value + d_val
            return target_mv_for_bumpless - (p_val + d_val)

        elif self.pid_structure == 'parallel_kp_global':
            p_val = current_error # P interne
            if self.derivative_action == 'on_pv' and self.kd_actual != 0 and self.previous_pv is not None:
                delta_pv = pv - self.previous_pv
                d_val = -self.kd_actual * delta_pv # D interne
            elif self.derivative_action == 'on_error' and self.kd_actual != 0 and self.previous_error is not None:
                delta_error = current_error - self.previous_error
                d_val = self.kd_actual * delta_error # D interne
            
            # target_mv = Kp_actual * (p_val + integral_term_value + d_val)
            # integral_term_value = (target_mv / Kp_actual) - p_val - d_val
            if self.kp_actual != 0:
                return (target_mv_for_bumpless / self.kp_actual) - (p_val + d_val)
            else: # Si Kp=0, l'intégrale ne peut pas être calculée pour atteindre une target_mv non nulle
                return 0
        return 0


    def set_active_state(self, active, sp_at_transition, pv_at_transition, mv_real_at_transition):
        state_changed = (self.is_active != active)
        if not state_changed:
            self.is_active = active # S'assurer que l'état est à jour même s'il ne change pas
            return

        if active and not self.is_active: # Passage Inactif -> Actif
            self.is_active = True
            self.integral_term = self._calculate_integral_for_bumpless(sp_at_transition, pv_at_transition, mv_real_at_transition)
            self.mv = self._limit_mv(mv_real_at_transition)
            self.last_active_mv = self.mv
            self.previous_pv = pv_at_transition
            
            current_error_at_transition = sp_at_transition - pv_at_transition
            if not self.direct_action: current_error_at_transition = -current_error_at_transition
            self.previous_error = current_error_at_transition
            logger.info(f"PID Simulé: Inactif -> Actif. MV={self.mv:.3f}, I_term={self.integral_term:.3f}")
        elif not active and self.is_active: # Passage Actif -> Inactif
            self.is_active = False
            self.last_active_mv = self.mv # Sauvegarder la dernière MV active
            # self.integral_term est déjà la valeur correcte du dernier pas actif
            logger.info(f"PID Simulé: Actif -> Inactif. MV gelée={self.last_active_mv:.3f}, I_term gelé={self.integral_term:.3f}")
        
        # self.is_active = active # Fait au début des branches


    def update(self, sp, pv):
        if self.previous_pv is None: self.previous_pv = pv
        
        current_error = sp - pv
        if not self.direct_action: current_error = -current_error

        if self.previous_error is None: self.previous_error = current_error

        if not self.is_active:
            self.mv = self.last_active_mv
            self.previous_pv = pv
            self.previous_error = current_error 
            return self.mv

        # --- Calcul des composants P, I, D ---
        # Terme Proportionnel
        if self.pid_structure == 'parallel_isa':
            self.p_component = self.kp_actual * current_error
        elif self.pid_structure == 'parallel_kp_global':
            self.p_component = current_error # Erreur directe, Kp appliqué plus tard

        # Terme Dérivé
        self.d_component = 0.0
        if self.kd_actual != 0: # kd_actual est Kp*Td/Ts (ISA) ou Td/Ts (Kp_global)
            if self.derivative_action == 'on_pv':
                delta_pv = pv - self.previous_pv
                self.d_component = -self.kd_actual * delta_pv 
            elif self.derivative_action == 'on_error':
                delta_error = current_error - self.previous_error
                self.d_component = self.kd_actual * delta_error

        # Terme Intégral (mise à jour)
        # self.integral_term est l'accumulateur qui représente le terme I complet pour ISA,
        # ou l'accumulateur interne (somme de (Ts/Ti)*erreur) pour Kp_global.
        # ki_actual est Kp*Ts/Ti (ISA) ou Ts/Ti (Kp_global)
        if self.ki_actual != 0:
            self.integral_term += self.ki_actual * current_error
        
        self.i_component = self.integral_term # Stocker pour clarté/debug

        # --- Assemblage de la sortie MV ---
        mv_candidate = 0.0
        if self.pid_structure == 'parallel_isa':
            mv_candidate = self.p_component + self.i_component + self.d_component
        elif self.pid_structure == 'parallel_kp_global':
            sum_internal_terms = self.p_component + self.i_component + self.d_component
            mv_candidate = self.kp_actual * sum_internal_terms
        
        current_mv_limited = self._limit_mv(mv_candidate)

        # --- Anti-Windup (Back-Calculation) ---
        if mv_candidate != current_mv_limited and self.ki_actual != 0 :
            if self.pid_structure == 'parallel_isa':
                # Pour ISA, self.integral_term EST le terme I.
                # On ajuste directement self.integral_term.
                self.integral_term = current_mv_limited - (self.p_component + self.d_component)
            elif self.pid_structure == 'parallel_kp_global' and self.kp_actual != 0:
                # Pour Kp_global, self.integral_term est l'accumulateur I interne.
                # On doit trouver la nouvelle valeur de cet accumulateur.
                # mv_lim = Kp * (P_int + I_int_new + D_int) => I_int_new = (mv_lim / Kp) - P_int - D_int
                self.integral_term = (current_mv_limited / self.kp_actual) - \
                                     (self.p_component + self.d_component)
            # logger.debug(f"Anti-windup: MV_cand={mv_candidate:.2f}, MV_lim={current_mv_limited:.2f}, new I_term={self.integral_term:.2f}")
        
        self.mv = current_mv_limited
        self.previous_pv = pv
        self.previous_error = current_error
        self.last_active_mv = self.mv

        return self.mv