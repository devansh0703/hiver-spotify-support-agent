# Eval summary

**agent** n=207 failed=0
escalation acc=0.9807 prec=1.0 rec=0.7895 f1=0.8824 (tp/fp/fn/tn=15/0/4/188, rate=0.0725)
intent acc=0.7391  reply_presence=1.0

**trivial** n=207
escalation acc=0.9082 prec=0.0 rec=0.0 f1=0.0 (tp/fp/fn/tn=0/0/19/188, rate=0.0)
intent acc=0.1884  reply_presence=1.0

**simple** n=207
escalation acc=0.8019 prec=0.0769 rec=0.1053 f1=0.0889 (tp/fp/fn/tn=2/24/17/164, rate=0.1256)
intent acc=0.0  reply_presence=0.8841

**judge** n=207
  helpfulness: mean=3.974 (n=192)
  voice: mean=4.88 (n=192)
  groundedness: mean=4.839 (n=192)
  escalation_decision: mean=4.981 (n=207)
  overall: mean=4.703 (n=207)
  pct overall>=4: 0.971

**judge-human agreement** {"n": 29, "judge_endorses_agent_decision_where_it_matches_gold": 0.9655, "n_divergent_from_gold_in_subset": 0, "judge_scores_on_divergent_rows": [], "escalation_judge_vs_human": {"n": 29, "exact_match": 0.931, "mean_abs_diff": 0.138, "note": "human prevalence skew makes kappa degenerate; diff is the informative stat"}, "helpfulness_exact_match": 0.25, "helpfulness_mean_abs_diff": 0.792, "helpfulness_binarized_kappa": 0.5}

**top intent confusions (agent)**
  gold=feature_request_feedback -> pred=technical_issue (5)
  gold=feature_request_feedback -> pred=content_library (5)
  gold=feature_request_feedback -> pred=device_integration (5)
  gold=billing_subscription -> pred=account_access (4)
  gold=device_integration -> pred=technical_issue (3)
  gold=billing_subscription -> pred=account_admin (3)
  gold=technical_issue -> pred=device_integration (3)
  gold=account_access -> pred=device_integration (2)
  gold=account_admin -> pred=technical_issue (2)
  gold=closing_acknowledgement -> pred=other_offtopic (2)
  gold=technical_issue -> pred=content_library (2)
  gold=account_admin -> pred=account_access (1)
