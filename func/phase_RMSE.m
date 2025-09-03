function re = phase_RMSE(x_es,x_gt)

re = norm2(x_es - x_gt) / norm2(x_gt);

function val = norm2(x)
    val = norm(x(:),2);
end

end

