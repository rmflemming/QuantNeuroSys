function prediction = continuous_decoder(coefficients,firing_rates)
    prediction = [coefficients*firing_rates']';
end

