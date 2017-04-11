% BioE 1585 - homework 4, question 3

function im = retina(pct, sd, image)
    imSize = max(size(image));
    rfSize = round(pct * imSize);
    dog = fspecial('log',rfSize, rfSize*sd); %Laplacian of Gaus filter, size rfSize
    c = -filter2(dog, image); % switch the sign here for an off-center cell
    % these two lines rescale c between 0 and 1
    c = c - min(min(c)); 
    c = c/max(max(c));
    keyboard    % uncomment this to explore how the code behaves.
    im = c;
    imshow(im)

end
