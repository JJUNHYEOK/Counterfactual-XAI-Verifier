function analysis = analyze_results(result)
    analysis.requirement_satisfied = result.collisionFlag == 0 && result.minDistance > 2.0;
    analysis.failure_type = result.failureType;
    analysis.risk_score = result.riskScore;
    analysis.confidence_trend = result.confidenceTrend;
    analysis.miss_rate_trend = result.missRateTrend;
    analysis.summary = "Fog and low illumination caused late obstacle avoidance.";
end