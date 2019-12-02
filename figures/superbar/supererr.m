%SUPERERR  Plot error bars in X and Y directions.
%   SUPERERR(X,Y,XE,YE) plots error bars at co-ordinates X(i),Y(i) with
%   lower error bounds XE(i,1) and YE(i,1) and upper error bounds XE(i,2)
%   and YE(i,2). By default, if XE or YE has only a single column, the
%   same length is used for both upper and lower bounds. If XE or YE
%   contains only 1 or 2 elements (and X contains more than 2 elements)
%   the same bounds are used for all datapoints.
%
%   SUPERERR(X,Y,XE) or SUPERERR(X,Y,XE,[]) or SUPERERR(X,Y,XE,NaN) plots
%   error bars in the X-direction only.
%
%   SUPERERR(X,Y,[],YE) or SUPERERR(X,Y,NaN,YE) plots error bars in the
%   Y-direction only.
%
%   Various error bar types, and colors may be obtained with
%   SUPERERR(X,Y,XE,YE,S) where S is a character string made from one
%   element each from any or all the following 2 columns:
%
%          b     blue          I     two-direction capped error-bar
%          g     green         |     two-direction stave without caps
%          r     red           T     one-direction capped error-bar
%          c     cyan          '     one-direction error-bar without caps
%          m     magenta       =     two-direction caps without stave
%          y     yellow        _     one-direction cap without stave
%          k     black
%          w     white
%
%   For one-directional error bar styles, a vector input for XE or YE is
%   interpretted as an upper bound only. If XE or YE is a two-column
%   input, an error bar is shown in each direction. The default style
%   is 'I'.
%
%   Different error bar styles in X and Y directions can be obtained with
%   SUPERERR(X,Y,XE,YE,XS,YS), where XS is the X error bar style, and YS
%   is the Y error bar style.
%
%   SUPERERR(...,W) with or without error bar style specified, will use
%   error bars with caps of width W.
%
%   The main inputs can be followed by parameter/value pairs to specify
%   additional properties of the lines.
%
%   Colors of error bars can be specified with RGB triples, passed as a
%   key/parameter pair. Color parameters can be either an n-by-1 char
%   specifying one of the colorspec values listed above, or an n-by-3
%   RGB triple. The number of colors specified need not match the number
%   of error bars; if fewer colors are provided, they are looped over
%   cyclically. The keys 'Color', 'XColor', and 'YColor' can be used to
%   specify the color of error bars for both directions, X-direction
%   only, and Y direction only, respectively.
%
%   SUPERERR(AX,...) plots into the axes with handle AX instead of GCA.
%
%   H = SUPERERR(...) returns a matrix output with handles to the error
%   bars, which are line objects. The first column contains error bars
%   in the X-direction, and the second column the Y-direction. Omitted
%   bars correspond to NaN values in H.
%
%   Please note that only linear-linear axes are supported. Other
%   packages are available if you want to plot error bars on log-log or
%   log-linear axes.

% Copyright (c) 2016,  Scott C. Lowe <scott.code.lowe@gmail.com>
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

function varargout = supererr(X, Y, XE, YE, X_style, Y_style, cap_width, ...
    varargin)

VALID_STYLES = 'I|T''=_';
COLOR_SPECS = 'rgbwcmyk';
DEFAULT_STYLE = 'I';

% Check number of inputs is okay
if exist('narginchk', 'builtin')
    narginchk(3, Inf);
elseif abs(nargin) < 3
    error('MATLAB:narginchk:notEnoughInputs', 'Not enough input arguments.');
end

% Extend the reach of varargin
if nargin>=7
    varargin = [{cap_width}, varargin];
end
if nargin>=6
    varargin = [{Y_style}, varargin];
end
if nargin>=5
    varargin = [{X_style}, varargin];
end
if nargin>=4
    varargin = [{YE}, varargin];
end
% Must be at least 3 input arguments
varargin = [{X, Y, XE}, varargin];

% Strip out axes input if it is there
[ax, varargin, nargs] = axescheck(varargin{:});
if ~isempty(ax)
    % We can't pass an axes argument to the line function because it only
    % became available in R2016a, so instead we change axes if necessary.
    % Make a cleanup object to revert focus back to previous axes
    prev_ax = gca();
    finishup = onCleanup(@() axes(prev_ax));
    % Change focus to the axes we want to work on
    axes(ax);
end
% Check number of inputs is still okay
if nargs<3
    error('Must provide at least 3 input arguments, in addition to axes.');
end
% Read out parameters with ax potentially removed from args
X = varargin{1};
Y = varargin{2};
XE = varargin{3};

% Default inputs for XE and YE
if isempty(XE)
    XE = nan(numel(X), 1);
end
if nargs<4 || isempty(varargin{4})
    YE = nan(numel(X), 1);
else
    YE = varargin{4};
end

% Just keep the rest of the arguments
varargin = varargin(5:end);

% Default args
X_style = '';
Y_style = '';
cap_width = [];

% Function to check that style is valid
function tf = check_is_style(t)
    if numel(t)==1
        tf = ismember(t, VALID_STYLES) || ismember(t, COLOR_SPECS);
    elseif numel(t)==2
        tf = (...
            ismember(t(1), VALID_STYLES) && ismember(t(2), COLOR_SPECS) ...
            ) || (...
            ismember(t(2), VALID_STYLES) && ismember(t(1), COLOR_SPECS) ...
            );
    else
        tf = false;
    end
end
% Parse the style and cap_width components out of varargin
numeric_in = 0;
string_in = 0;
for iArg=1:min(3, numel(varargin))
    if isnumeric(varargin{iArg})
        if numeric_in>0
            break;
        end
        cap_width = varargin{iArg};
        numeric_in = numeric_in + 1;
    elseif ischar(varargin{iArg}) && ...
            (isempty(varargin{iArg}) || check_is_style(varargin{iArg}))
        if string_in==0
            X_style = varargin{iArg};
        elseif string_in==1
            Y_style = varargin{iArg};
        else
            break;
        end
        string_in = string_in + 1;
    else
        break;
    end
end
% Cut away the style and cap_width inputs
varargin = varargin((numeric_in + string_in + 1):end);

% Default inputs
if isempty(cap_width)
    cap_width = 0.5;
end
if isempty(X_style)
    X_style = DEFAULT_STYLE;
end
if isempty(Y_style)
    Y_style = X_style;
end

% Check inputs
assert(ischar(X_style), 'X_style must be a char string.');
assert(ischar(Y_style), 'Y_style must be a char string.');
% Set default colors
X_color = [.2 .2 .2];
Y_color = [.2 .2 .2];
% Check if style contains color spec
if ~isempty(X_style)
    idx = find(ismember(X_style, VALID_STYLES), 1, 'last');
    if isempty(idx)
        X_color = X_style;
        X_style = DEFAULT_STYLE;
    elseif numel(idx) < numel(X_style)
        X_color = X_style([1:idx-1, idx+1:end]);
        X_style = X_style(idx);
    end
end
if ~isempty(Y_style)
    idx = find(ismember(Y_style, VALID_STYLES), 1, 'last');
    if isempty(idx)
        Y_color = Y_style;
        Y_style = DEFAULT_STYLE;
    elseif numel(idx) < numel(Y_style)
        Y_color = Y_style([1:idx-1, idx+1:end]);
        Y_style = Y_style(idx);
    end
end
% Check if varargin contains colors
is_color_arg = false(size(varargin));
for iArg = 1:numel(varargin)-1
    if isequal(varargin{iArg}, 'Color')
        is_color_arg(iArg:(iArg+1)) = true;
        X_color = varargin{iArg+1};
        Y_color = varargin{iArg+1};
    elseif isequal(varargin{iArg}, 'XColor')
        is_color_arg(iArg:(iArg+1)) = true;
        X_color = varargin{iArg+1};
    elseif isequal(varargin{iArg}, 'YColor')
        is_color_arg(iArg:(iArg+1)) = true;
        Y_color = varargin{iArg+1};
    end
end
% Remove color arguments from varargin
varargin = varargin(~is_color_arg);

% Replicate single inputs so there is a copy for every error bar
if numel(XE)==1
    XE = XE * ones(numel(X), 1);
end
if numel(YE)==1
    YE = YE * ones(numel(X), 1);
end
if numel(XE)==2 && (numel(X)~=2 || size(XE,1)==1)
    XE = repmat(XE(:)', numel(X), 1);
end
if numel(YE)==2 && (numel(X)~=2 || size(YE,1)==1)
    YE = repmat(YE(:)', numel(X), 1);
end
% Reshape colour arguments
if ischar(X_color)
    X_color = X_color(:);
else
    assert(size(X_color, ndims(X_color))==3, ...
        'Last dimension must be RGB channel.')
    X_color = reshape(X_color, numel(X_color)/3, 3);
end
if ischar(Y_color)
    Y_color = Y_color(:);
else
    assert(size(Y_color, ndims(Y_color))==3, ...
        'Last dimension must be RGB channel.')
    Y_color = reshape(Y_color, numel(Y_color)/3, 3);
end

% Check inputs are okay
assert(numel(X)==numel(Y), 'X and Y need to be the same size.');

check_e = @(t) (isempty(t) || ...
    (size(t,1)==numel(X) && ismatrix(t) && (size(t,2)==1 || size(t,2)==2)));
assert(check_e(XE), 'Input XE is badly formatted.');
assert(check_e(YE), 'Input YE is badly formatted.');
assert(all(XE(:)>=0 | isnan(XE(:))), 'Input XE must be non-negative');
assert(all(YE(:)>=0 | isnan(YE(:))), 'Input YE must be non-negative');

assert(isnumeric(cap_width), 'Cap width must be a double.');
assert(numel(cap_width)==1, 'Input cap_width must be a scalar.');

assert(ischar(X_style), 'X_style must be a string');
assert(ischar(Y_style), 'Y_style must be a string');
assert(all(ismember(X_style, VALID_STYLES)), ...
    'X_style must be one of %s', VALID_STYLES);
assert(all(ismember(Y_style, VALID_STYLES)), ...
    'Y_style must be one of %s', VALID_STYLES);

% Main
H = nan(numel(X), 2);
for i = 1:numel(X)
    % Include co-ordinates for X error bar
    if ~all(isnan(XE(i,:)))
        % Take the x and y co-ordinates the other way around, since this is
        % an x errobar
        [yco, xco] = compute_errorbar_relative_coordinates(...
            XE(i,:), X_style, cap_width);
        % Shift co-ordinates to centre at the correct location
        xco = xco + X(i);
        yco = yco + Y(i);
        % Let the color loop around if it runs out
        iCol = 1 + mod(i-1, size(X_color, 1));
        % Draw the errorbar with line
        H(i, 1) = line(xco, yco, 'Color', X_color(iCol, :), varargin{:});
    end
    % Include co-ordinates for Y error bar
    if ~all(isnan(YE(i,:)))
        [xco, yco] = compute_errorbar_relative_coordinates(...
            YE(i,:), Y_style, cap_width);
        % Shift co-ordinates to centre at the correct location
        xco = xco + X(i);
        yco = yco + Y(i);
        % Let the color loop around if it runs out
        iCol = 1 + mod(i-1, size(Y_color, 1));
        % Draw the error bar with line
        H(i, 2) = line(xco, yco, 'Color', Y_color(iCol, :), varargin{:});
    end
end

if nargout==0
    varargout = {};
else
    varargout = {H};
end

end


%compute_errorbar_relative_coordinates
%   Computes the co-ordinates of an error bar in the y-direction, centred
%   around (0,0) with style given by the input style. E input is the error
%   on the value, which can be either a single value or two values. In the
%   former case, the error bar can be either drawn in both directions the
%   same length or only one. If E contains two values, the error bar is
%   always drawn in both directions, with different lengths.
function [xx, yy] = compute_errorbar_relative_coordinates(E, style, width)

style = upper(style);

xx = [];
yy = [];

% Add co-ordinates for the stave
if numel(E)>1 && ismember(style, 'I|T''')
    % Asymmetric stave 
    xx = [xx,     0,    0, NaN];
    yy = [yy, -E(1), E(2), NaN];
elseif numel(E)==1 && ismember(style, 'I|')
    % Symmetric stave
    xx = [xx,  0, 0, NaN];
    yy = [yy, -E, E, NaN];
elseif numel(E)==1 && ismember(style, 'T''')
    % Half stave
    xx = [xx, 0, 0, NaN];
    yy = [yy, 0, E, NaN];
end
% Don't draw caps on if they don't have any width
if all(width==0)
    return;
end
% Duplicate width if there is only one input
if numel(width)==1
    width = [1, 1] * width;
end
% Halve the width, so we have the amount on each side
width = width / 2;
% Add co-ordinates for the caps
if numel(E)>1 && ismember(style, 'IT=_')
    % Asymmetric caps
    xx = [xx, -width(1), width(1), NaN, -width(2), width(2), NaN];
    yy = [yy,     -E(1),    -E(1), NaN,      E(2),     E(2), NaN];
elseif numel(E)==1 && ismember(style, 'I=')
    % Symmetric caps
    xx = [xx, -width(1), width(1), NaN, -width(2), width(2), NaN];
    yy = [yy,        -E,       -E, NaN,         E,        E, NaN];
elseif numel(E)==1 && ismember(style, 'T_')
    % Single cap
    xx = [xx, -width(2), width(2), NaN];
    yy = [yy,         E,        E, NaN];
end

end