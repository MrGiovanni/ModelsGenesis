%SUPERBAR  Plots bar graph with errors and p-values indicated
%   SUPERBAR(X,Y) draws the columns of the M-by-N matrix Y as M groups of
%   N vertical bars with location given by X and length given by Y. The
%   vector X should not have duplicate values.
%
%   SUPERBAR(Y) uses the default value of X=1:M. For vector inputs,
%   SUPERBAR(X,Y) or SUPERBAR(Y) draws LENGTH(Y) bars.
%
%   SUPERBAR(...,'E',E) plots the error, E, on each bar using SUPERERR. E
%   can be a matrix with the same number of elements as Y to specify
%   symmetric or single-directional errors, or an array with twice as many
%   elements (sized M-by-N-by-2 for instance) to specify asymmetric
%   errorbars. If E contains only 1 or 2 elements, the same symmetric or
%   asymmetric error bounds are used for every bar. Note that the ambiguous
%   case of plotting 2 bars with a/symmetric error bars should be
%   disambiguated by using a 1-by-1-by-2 array to apply the same asymmetric
%   error bounds to both bars and a 1-by-2 or 2-by-1 array for two
%   different but symmetric errors.
%
%   SUPERBAR(...,'P',P) with P the same size as Y adds stars above bars to
%   indicated whether the p-values in P are significant. Significance
%   thresholds can be set with the 'PStarThreshold' parameter (see below).
%
%   SUPERBAR(...,'P',P) with P a symmetric matrix sized (N*M)-by-(N*M) adds
%   lines between each bar indicating comparisons between them, with stars
%   above the lines which correspond to significant comparisons. Unmeasured
%   comparisons can be indicated with NaN values in P. This option is only
%   available for vertically oriented bars.
%
%   SUPERBAR(AX,...) plots into the axes with handle AX instead of GCA.
%
%   The inputs can be followed by parameter/value pairs to specify
%   additional properties, as follows.
%
%       General plot attributes
%       -----------------------
%       'Orientation' : Orientation of the bar plot. Set to 'v' for
%           vertical bars, or 'h' for horizontal bars. Default is 'v'. Note
%           that X is still the location of the bars and Y the length of
%           the bars, even if orientation 'h' is used.
%
%       Bar attributes
%       --------------
%       'BaseValue' : Base value from which bars begin. Default is 0.
%       'BarWidth' : Width of bars. Default is 80% of the minimum
%           separation between bars as specified in X.
%       'BarRelativeGroupWidth' : Relative width of bars when they are
%           grouped. Setting BarRelativeGroupWidth to 1 will have no spaces
%           between bars. Default is 0.8.
%       'BarFaceColor' : Color of the bars. Can be 'none', a colorspec
%           string (one of 'rgbymckw'), or an RGB array, or a cell array of
%           colorspec strings, RGB arrays, and 'none' strings. In the case
%           of an RGB array, the size can be m-by-3 or m-by-n-by-3. If the
%           input contains fewer than M rows (or N columns in the later
%           case), the colors are repeated cyclically. Similarly, a cell
%           array or char array input can be m-by-n to specify multiple
%           colours, and the array will be repeated as necessary. The
%           input can be set to 'none' for all transparent bar faces. Cell
%           array inputs may contain 'none' for some cell elements to make
%           only certain bars have transparent faces. Default is
%           [.4, .4, .4].
%       'BarEdgeColor' : Color of the bars edges. For input options, see
%           'BarFaceColor'. Default is 'none', unless 'BarFaceColor' was
%           set to 'none', in which case 'BarEdgeColor' is [.4 .4 .4].
%       'BarLineWidth' : Linewidth for the bar edges. Default is 2.
%
%       Errorbar attributes
%       -------------------
%       'E' : Errorbar magnitudes. Can be the same size as Y for specifying
%           symmetric or one-sided errorbars, or M-by-N-by-2 for asymmetric
%           errorbars. If E contains only a single value or two values, the
%           same symmetric or asymmetric errorbars are used for each bar.
%           Note that the ambiguous case with two bars and two errorbar
%           values should be disambiguated by ensuring E is 3-dimensional
%           when specifying asymmetric error bounds. If empty, no errorbars
%           are shown. Default is [].
%       'ErrorbarRelativeWidth' : Width of the errorbar caps, relative to
%           the bar width. Default is 0.5.
%       'ErrorbarColor' : Color of the errorbars. For input options, see
%           'BarFaceColor'. Default is the same as BarEdgeColor if it is
%           not 'none', otherwise 0.7 * BarFaceColor.
%       'ErrorbarStyle' : Shape of the errorbars to plot. Different
%           combinations allow plotting only stave, only caps, only
%           errorbars in a single direction, etc. Default is 'I', which has
%           a stave and cap in both directions always. See SUPERERR for a
%           list of possible errorbar styles. For instance, single-
%           directional errorbars can be acheived with the 'T' style.
%       'ErrorbarLineWidth' : LineWidth for errorbar lines. Default is 2.
%
%       P-value comparison attributes
%       -----------------------------
%       'P' : P-values. Can be either the same size as Y for specifying the
%           significance of each bar, or an (N*M)-by-(N*M) symmetric matrix
%           to indicate comparisons between each bar. If empty, no stars or
%           comparison lines are shown. Default is [].
%       'PStarThreshold' : Values which p-values must exceed (be smaller
%           than or equal to) to earn a star. If PStarShowGT is false,
%           significance is indicated with a star for every value in
%           PStarThreshold which exceeds the p-value. If PStarShowGT is
%           true, a p-value smaller than every element in PStarThreshold is
%           indicated with (e.g.) '>***' instead of '****', to show the
%           maximum measured precision has been exceeded. Default is
%           [0.05, 0.01, 0.001, 0.0001].
%       'PStarLatex' : Whether to use LaTeX for the significance text. If
%           set to 'off', no text will use the LaTeX interpreter. If
%           'auto', the stars for significance such as '*' and '>**' will
%           be wrapped in '$' symbols and the interpreter set to LaTeX
%           mode, but text indicating non-significance will be left
%           unchanged. NB: Using 'auto' will cause your text label fonts to
%           be inconsistent with each other. If 'all' or 'on', text
%           indicating non-significance will also use the LaTeX
%           interpreter. NB: Using 'all' or 'on' may cause the font for
%           'n.s' text to be different to your axes font. Default is 'off'.
%       'PStarIcon' : Character code to use for stars. Suggested values
%           include the following values:
%               '*' : Asterisk, U+002A
%               char(8727) : Asterisk operator, U+2217
%               char(10033) : Heavy asterisk, U+2731
%               char(10035) : Eight-spoked asterisk, U+2733
%               char(128944) : Five-spoked asterisk, U+1F7B0
%           These integers are each the decimal version of the unicode code
%           point for the characters, such as would be returned by
%           `hex2dec('2217')` for instance, where U+2217 is the
%           hexadecimal code point for asterisk operator. If PStarLatex is
%           disabled, the default is char(8727), the asterisk operator. If
%           PStarLatex is enabled, the default value is '*'. When using
%           PStarLatex mode, PStarIcon must be a character recognised by
%           LaTeX (a string composed of ASCII-only characters). When
%           PStarLatex mode is disabled, using a literal asterisk, '*', is
%           possible but not advised since the regular asterisk is in the
%           position of a raised character, and will therefore not sit in
%           the middle of its bounding box. If you see boxes showing a
%           rectangle with a cross through them or the wrong symbol instead
%           of a star, this is because your font does not include this
%           symbol.
%       'PStarColor' : Color of the text for significance stars. Default is
%           [.2 .2 .2].
%       'PStarBackgroundColor' : Background color of the text. Default is
%           the axes background color for paired comparisons, and 'none'
%           for single bar p-values.
%       'PStarFontSize' : Font size of the text for significance stars.
%           Default is 14.
%       'PStarShowNS' : Whether to write 'n.s.' above comparisons which are
%           not significant. Default is true.
%       'PStarShowGT' : Whether to show a greater-than sign (>) for
%           p-values which are smaller than every value in PStarThreshold.
%           Default is true.
%       'PStarOffset' : Distance of the stars from the top of the errorbars
%           (or bars if no errorbars used). Default is 5% of the tallest
%           bar for single comparisons, or a third of PLineOffset for
%           paired comparisons.
%       'PStarFixedOrientation' : Whether to always show stars in the
%           normal reading direction. If false, the stars will be rotated
%           for horizontal bars. Default is true for single bar p-values
%           and false for pairwise comparisons.
%       'PLineColor' : Color of the lines indicating comparisons between
%           bars. Default is [.5 .5 .5].
%       'PLineWidth' : Width of the lines indicating comparisons between
%           bars. Default is 2.
%       'PLineOffset' : Vertical space between the comparison lines.
%           Default is 10% of the tallest bar.
%       'PLineOffsetSource' : Vertical space between the comparison lines,
%           used only when there is no horizontal spaces between lines
%           coming up from the same bar. Default is a quarter of
%           PLineOffset.
%       'PLineSourceRelativeSpacing' : Maximum space between each line
%           coming from the top of a bar, relative to the width of the bar.
%           Default is 0.5.
%       'PLineSourceRelativeBreadth' : Maximum space which the lines coming
%           from each bar can collectively occupy, relative to the width of
%           the bar. Default is the same as ErrorbarRelativeWidth, if it is
%           non-zero, otherwise 0.8.
%       'PLineBacking' : Whether to pad p-value comparison lines by
%           plotting them on top of a backing line. Default is false if
%           there is no space between the line sources (i.e. one of
%           PLineSourceRelativeSpacing and PLineSourceRelativeBreadth is
%           equal to 0) and true otherwise.
%       'PLineBackingWidth' : Width of the line giving a backing color
%           behind each the comparison line. Default is 3 times PLineWidth,
%           so that the space on each side of the line is the same width as
%           the line itself.
%       'PLineBackingColor' : Color to use for the backing behind
%           comparison lines. Default is the axes background color.
%
%   [HB, HE, HPT, HPL, HPB] = SUPERBAR(...) returns handles to the
%   generated graphics objects. HB contains handles to the bars themselves,
%   in a matrix whose size matches that of Y. HE contains handles to the
%   errorbars, in a matrix whose size matches that of Y. HPT contains
%   handles to the text showing p-value siginficance levels with stars. HPL
%   contains handles to the comparison lines between bars. HPB contains
%   handles to the background behind the comparison lines.
%
%   Note that unlike BAR and BARH, bars plotted with SUPERBAR are always
%   grouped and never stacked.
%
%   FAQs
%   ----
%   - If you see boxes showing a rectangle with a cross through them
%     instead of stars, this is because your font does not include the
%     symbol being used. If you're using the default settings and want a
%     quick fix, set PStarLatex to 'on'. For a proper fix, you'll have to
%     install a copy of that font with this symbol included (if possible)
%     and get MATLAB to use it (which can be tricky, to say the least).
%
%   See also SUPERERR, BAR, BARH.

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

function varargout = superbar(X, Y, varargin)

% Check number of inputs is okay
if exist('narginchk', 'builtin')
    narginchk(1, Inf);
elseif abs(nargin) < 1
    error('MATLAB:narginchk:notEnoughInputs', 'Not enough input arguments.');
end

% Extend the reach of varargin
if nargin>=2
    varargin = [{Y}, varargin];
end
varargin = [{X}, varargin];

% Strip out axes input if it is there
[ax, varargin, nargs] = axescheck(varargin{:});
ax_or_empty = ax;
% Otherwise, default with the current axes
if isempty(ax)
    ax = gca;
end
% Check number of inputs is still okay
if nargs<1
    error('Must provide at least 1 input argument, in addition to axes.');
end

% Input handling for X and Y
if nargs==1 || ischar(varargin{2})
    % Deal with omitted X input
    Y = varargin{1};
    X = [];
    % Drop the argument
    varargin = varargin(2:end);
else
    % Take X and Y out of varargin
    X = varargin{1};
    Y = varargin{2};
    % Drop these arguments
    varargin = varargin(3:end);
end
if ~ismatrix(Y)
    error('Y should have no more than 2-dimensions (%d given).', ndims(Y));
end
if size(Y, 1)==1
    Y = Y';
end
if isempty(X)
    X = (1:size(Y, 1))';
end
if size(X, 1)~=size(Y, 1) && size(X, 2)==size(Y, 1)
    X = X';
end
if size(X, 1)~=size(Y, 1)
    error('X and Y must be the same size in dimension 1 (%d and %d given)', ...
        size(X, 1), size(Y, 1));
end

% Use parser for the rest of the arguments
parser = inputParser;
% Make a function for adding arguments to parser which works for both new
% and old matlab versions
function safelyAddParameter(p, varargin)
    if ismethod(p, 'addParameter');
        % Added in R2013b
        p.addParameter(varargin{:});
    elseif ismethod(p, 'addParamValue')
        % Added in R2007a, and deprecated from R2013b onwards
        p.addParamValue(varargin{:});
    else
        error(...
            ['Could not find a method to add parameters to the input' ...
            ' parser. Please upgrade your MATLAB installation to R2013a' ...
            'or later.']);
    end
end
% Plot attributes
safelyAddParameter(parser, 'Orientation', 'v', ...
    @ischar);
safelyAddParameter(parser, 'BaseValue', 0, ...
    @(t) (isscalar(t)) && isnumeric(t));
% Bar attributes
safelyAddParameter(parser, 'BarWidth', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'BarRelativeGroupWidth', 0.8, ...
    @(t) (isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'BarFaceColor', []);
safelyAddParameter(parser, 'BarEdgeColor', []);
safelyAddParameter(parser, 'BarLineWidth', 2);
% Errorbar attributes
safelyAddParameter(parser, 'E', []);
safelyAddParameter(parser, 'ErrorbarRelativeWidth', 0.5, ...
    @(t) (isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'ErrorbarColor', []);
safelyAddParameter(parser, 'ErrorbarStyle', 'I', ...
    @ischar);
safelyAddParameter(parser, 'ErrorbarLineWidth', 2, ...
    @(t) (isscalar(t)) && isnumeric(t));
% P-value attributes
safelyAddParameter(parser, 'P', []);
safelyAddParameter(parser, 'PStarThreshold', [0.05, 0.01, 0.001, 0.0001], ...
    @isnumeric);
safelyAddParameter(parser, 'PStarLatex', 'off', ...
    @(t) (ischar(t) && ismember(t, {'off', 'auto', 'all', 'on'})));
safelyAddParameter(parser, 'PStarIcon', '');
safelyAddParameter(parser, 'PStarColor', [.2 .2 .2]);
safelyAddParameter(parser, 'PStarBackgroundColor', []);
safelyAddParameter(parser, 'PStarFontSize', 14, ...
    @isscalar);
safelyAddParameter(parser, 'PStarShowNS', true, ...
    @isscalar);
safelyAddParameter(parser, 'PStarShowGT', true, ...
    @isscalar);
safelyAddParameter(parser, 'PStarOffset', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PStarFixedOrientation', [], ...
    @isscalar);
safelyAddParameter(parser, 'PLineColor', [.5 .5 .5]);
safelyAddParameter(parser, 'PLineWidth', 2, ...
    @(t) (isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineOffset', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineOffsetSource', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineSourceRelativeSpacing', [], ...
    @(t) (isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineSourceRelativeBreadth', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineBacking', []);
safelyAddParameter(parser, 'PLineBackingWidth', [], ...
    @(t) (isempty(t) || isscalar(t)) && isnumeric(t));
safelyAddParameter(parser, 'PLineBackingColor', []);
% Parse the arguments
parse(parser, varargin{:});
input = parser.Results;

% Default input arguments which inherit values from others
% Bar defaults
if isempty(input.BarWidth)
    if numel(X)>1
        % Default with 0.8 of the smallest distance between bars
        input.BarWidth = 0.8 * min(diff(sort(X(:))));
    else
        % Just use 0.8 if there's only one bar
        input.BarWidth = 0.8;
    end
end
if isempty(input.BarFaceColor)
    if isempty(input.BarEdgeColor) || isequal(input.BarFaceColor, 'none')
        input.BarFaceColor = [.4, .4, .4];
    else
        input.BarFaceColor = 'none';
    end
end
if isempty(input.BarEdgeColor)
    if isequal(input.BarFaceColor, 'none')
        input.BarEdgeColor = [.4 .4 .4];
    else
        input.BarEdgeColor = 'none';
    end
end
% Errorbar defaults
if isempty(input.ErrorbarColor)
    % Try taking the colour from the errorbar style
    COLOR_SPECS = 'rgbwcmyk';
    isColorSpec = ismember(COLOR_SPECS, input.ErrorbarStyle);
    if any(isColorSpec)
        idx = find(isColorSpec, 1, 'last');
        input.ErrorbarColor = input.ErrorbarStyle(idx);
    elseif ~isequal(input.BarEdgeColor, 'none')
        % Try taking the color from the bar edge
        input.ErrorbarColor = fix_colors_cell(input.BarEdgeColor);
    elseif ~isequal(input.BarFaceColor, 'none')
        % Try taking the color from the bar face
        color = input.BarFaceColor;
        % Convert cell into RGB array
        color = fix_colors_char(color);
        % Convert char array into RGB array
        color = fix_colors_cell(color);
        % Ensure array is MxNx3
        siz = size(color);
        if length(siz)==2 && siz(2)==3
            color = permute(color, [1 3 2]);
        end
        % Make the bar colour lighter
        lighter_color = 1 - 0.7 * (1 - color);
        % Make the bar colour darker
        darker_color = 0.7 * color;
        % Check when to take lighter and when to take darker
        is_too_dark = repmat(sum(color, 3) < 0.65, [1 1 3]);
        input.ErrorbarColor = ...
            is_too_dark .* lighter_color + ~is_too_dark .* darker_color;
    else
        % Well if you want everything transparent you can have it, I
        % guess, though maybe this should be an error instead
        input.ErrorbarColor = 'none';
    end
    % Fix rogue NaN values from 'none' within cells (an unusual event)
    input.ErrorbarColor(isnan(input.ErrorbarColor)) = 0.5;
end

% Fix size of X, and Y
% Split up bars which are composed of groups
if size(X, 2)==1
    [X, Y, input.BarWidth] = bar2grouped(X, Y, input.BarWidth, ...
        input.BarRelativeGroupWidth);
end
% Check size of X and Y match
assert(isequal(size(X), size(Y)), ...
    'Sizes of X and Y must match. Sizes were %s and %s.', ...
    mat2str(size(X)), mat2str(size(Y)));

% Fix shape of E
if numel(input.E)==1
    input.E = repmat(input.E, numel(Y), 1);
elseif numel(input.E)==2*numel(Y)
    input.E = reshape(input.E, numel(Y), 2);
elseif numel(input.E)==2 && (numel(Y)~=2 || ~ismatrix(input.E))
    input.E = repmat(input.E(:)', numel(Y), 1);
elseif numel(input.E)==numel(Y)
    input.E = input.E(:);
elseif ~isempty(input.E)
    error(...
        ['E input must contain either the same number of values as Y' ...
        ' (for symmetric errorbars), or twice as many values (for' ...
        ' asymmetric errorbars), or a single value/pair of values (to' ...
        ' use the same error for every bar).']);
end

% P-value defaults
if isempty(input.PLineOffset)
    % Base the offset on the maximum of the bars
    input.PLineOffset = 0.1 * max(abs(Y(:)));
end
if isempty(input.PStarOffset)
    if numel(input.P)==numel(X) || numel(input.P)==numel(Y)
        % If we're just showing the stars and no lines, base the offset on
        % the maximum of the bars
        input.PStarOffset = 0.05 * max(abs(Y(:)));
    else
        % If we're showing comparison lines, make the stars be a little
        % above the lines so its clear to which they belong
        input.PStarOffset = input.PLineOffset / 3;
    end
end
if isempty(input.PLineOffsetSource)
    % Vertical offset between source of comparison line above another line,
    % used when there is no horizontal space between sources.
    input.PLineOffsetSource = input.PLineOffset / 3;
end
if isempty(input.PStarFixedOrientation)
    if numel(input.P)==numel(Y)^2
        % For pairwise comparisons
        input.PStarFixedOrientation = false;
    else
        % For single bar significance
        input.PStarFixedOrientation = true;
    end
end
if isempty(input.PLineSourceRelativeSpacing)
    if ~isempty(input.ErrorbarRelativeWidth) && input.ErrorbarRelativeWidth>0
        % The maximum space between any pair of lines should be a fraction
        % of the errorbar width, if possible.
        input.PLineSourceRelativeSpacing = 1/3 * input.ErrorbarRelativeWidth;
    else
        % Otherwise base it on the bar width
        input.PLineSourceRelativeSpacing = 1/3;
    end
end
if isempty(input.PLineSourceRelativeBreadth)
    if ~isempty(input.ErrorbarRelativeWidth) && input.ErrorbarRelativeWidth>0
        % The breadth of the space the lines come from should be the same
        % as the width of the errorbars, if possible
        input.PLineSourceRelativeBreadth = input.ErrorbarRelativeWidth;
    else
        % Otherwise base it on the bar width
        input.PLineSourceRelativeBreadth = 0.8;
    end
end
if strcmp(input.PStarLatex, 'on')
    % Set this as an alias
    input.PStarLatex = 'all';
end
if isempty(input.PStarIcon)
    if strcmp(input.PStarLatex, 'off')
        % With latex off, we use an asterisk operator unicode symbol.
        input.PStarIcon = char(8727);
    else
        % With latex on, we use an asterisk, which will be converted into
        % an asterisk operator by latex.
        input.PStarIcon = '*';
    end
end
if isempty(input.PStarBackgroundColor)
    if numel(input.P)==numel(X) || numel(input.P)==numel(Y)
        input.PStarBackgroundColor = 'none';
    else
        % Background color behind significance text
        input.PStarBackgroundColor = get(ax, 'Color');
    end
end
if isempty(input.PLineBackingColor)
    % Color to pad lines with
    input.PLineBackingColor = get(ax, 'Color');
end
if isempty(input.PLineBackingWidth)
    input.PLineBackingWidth = 3 * input.PLineWidth;
end
if isempty(input.PLineBacking)
    if input.PLineSourceRelativeSpacing == 0 || ...
            input.PLineSourceRelativeBreadth == 0
        input.PLineBacking = false;
    else
        input.PLineBacking = true;
    end
end

% Fix relative widths
errorbarWidth = input.ErrorbarRelativeWidth * input.BarWidth;
PLineSourceBreadth = input.PLineSourceRelativeBreadth * input.BarWidth;
PLineSourceSpacing = input.PLineSourceRelativeSpacing * input.BarWidth;

% Fix char array input for colours; convert to RGB array
function C = fix_colors_char(C)
    if ~ischar(C)
        return;
    end
    if numel(C)==1
        C = colorspec2rgb(C);
        return;
    end
    if strcmp(C, 'none')
        return;
    end
    siz = size(C);
    C_out = nan([siz 3]);
    for iCol = 1:siz(2)
        for iRow = 1:siz(1)
            C_out(iRow, iCol, :) = colorspec2rgb(C(iRow, iCol));
        end
    end
    C = C_out;
end
% Fix cellarray for color input
function C = fix_colors_cell(C)
    if ~iscell(C)
        return;
    end
    siz = size(C);
    assert(numel(siz)<3, 'Too many dimensions for cellarray C.');
    C_out = nan([siz 3]);
    for iCol = 1:siz(2)
        for iRow = 1:siz(1)
            if ischar(C{iRow, iCol}) && strcmp(C{iRow, iCol}, 'none')
                % Encode 'none' as NaN. We'll decode it later.
                C_out(iRow, iCol, :) = NaN;
            elseif ischar(C{iRow, iCol})
                C_out(iRow, iCol, :) = colorspec2rgb(C{iRow, iCol});
            elseif isnumeric(C{iRow, iCol}) && numel(C{iRow, iCol})==3
                C_out(iRow, iCol, :) = C{iRow, iCol};
            else
                error('Cell array must contain only strings and RGB vectors');
            end
        end
    end
    C = C_out;
end
% Extend colors to be per bar
function C = extend_colors(C)
    siz = size(Y);
    siz_ = size(C);
    if ~ischar(C)
        assert(length(siz_)<=3, 'Too many dimensions for C.');
        assert(siz_(end)==3, 'Must be RGB color in C with 3 channels.');
    end
    if length(siz_)==2 && ( ~ischar(C) || isequal(C, 'none') )
        C = permute(C, [1, 3, 2]);
    end
    siz_ = size(C);
    C = repmat(C, ceil(siz(1) / siz_(1)), ceil(siz(2) / siz_(2)));
    C = C(1:siz(1), 1:siz(2), :);
end
input.BarFaceColor = extend_colors(fix_colors_cell(input.BarFaceColor));
input.BarEdgeColor = extend_colors(fix_colors_cell(input.BarEdgeColor));
input.ErrorbarColor = extend_colors(fix_colors_cell(input.ErrorbarColor));

% Map NaN colours back to 'none'
function C = nan2none(C)
    if all(isnan(C))
        C = 'none';
    end
end

% Check if hold is already on
wasHeld = ishold(ax);
% If not, clear the axes and turn hold on
if ~wasHeld;
    cla(ax);
    hold(ax, 'on');
end;

nBar = numel(Y);
hb = nan(size(Y));
for iBar=1:nBar
    % Get indices to tell which row and column to take colour from
    [i, j] = ind2sub(size(Y), iBar);
    % Plot bar
    if strncmpi(input.Orientation, 'h', 1)
        hb(iBar) = barh(ax, X(iBar), Y(iBar), input.BarWidth);
    else
        hb(iBar) = bar(ax, X(iBar), Y(iBar), input.BarWidth);
    end
    % Colour it in correctly
    set(hb(iBar), ...
        'FaceColor', nan2none(input.BarFaceColor(i,j,:)), ...
        'EdgeColor', nan2none(input.BarEdgeColor(i,j,:)), ...
        'BaseValue', input.BaseValue, ...
        'LineWidth', input.BarLineWidth);
end
% Add errorbars
if isempty(input.E)
    he = [];
elseif strncmpi(input.Orientation, 'h', 1)
    % Horizontal errorbars
    he = supererr(ax, Y, X, input.E, [], input.ErrorbarStyle, ...
        errorbarWidth, ...
        'Color', input.ErrorbarColor, ...
        'LineWidth', input.ErrorbarLineWidth);
    he = reshape(he(:, 1), size(Y));
else
    % Vertical errorbars
    he = supererr(ax, X, Y, [], input.E, input.ErrorbarStyle, ...
        errorbarWidth, ...
        'Color', input.ErrorbarColor, ...
        'LineWidth', input.ErrorbarLineWidth);
    he = reshape(he(:, 2), size(Y));
end
% Add p-values
if isempty(input.P)
    % Do nothing
    hpt = [];
    hpl = [];
    hpb = [];
elseif numel(input.P)==numel(Y)
    % Add stars above bars
    hpt = plot_p_values_single(ax_or_empty, ...
        X, Y, input.E, input.P, ...
        input.Orientation, input.BaseValue, input.PStarThreshold, ...
        input.PStarOffset, input.PStarShowNS, input.PStarShowGT, ...
        input.PStarFixedOrientation, input.PStarIcon, input.PStarLatex, ...
        {'Color', input.PStarColor, ...
         'FontSize', input.PStarFontSize, ...
         'BackgroundColor', input.PStarBackgroundColor});
    hpl = [];
    hpb = [];
elseif numel(input.P)==numel(Y)^2
    % Add lines and stars between pairs of bars
    [hpt, hpl, hpb] = plot_p_values_pairs(ax_or_empty, ...
        X, Y, input.E, input.P, ...
        input.Orientation, input.PStarThreshold, input.PLineOffset, ...
        input.PLineOffsetSource, input.PStarOffset, input.PStarShowNS, ...
        input.PStarShowGT, PLineSourceSpacing, PLineSourceBreadth, ...
        input.PLineBacking, input.PStarFixedOrientation, input.PStarIcon, ...
        input.PStarLatex, ...
        {'Color', input.PLineColor, ...
         'LineWidth', input.PLineWidth}, ...
        {'Color', input.PLineBackingColor, ...
         'LineWidth' input.PLineBackingWidth}, ...
        {'Color', input.PStarColor, ...
         'FontSize', input.PStarFontSize, ...
         'BackgroundColor', input.PStarBackgroundColor});
else
    error('Bad number of P-values');
end

% If hold was off, turn it off again
if ~wasHeld; hold(ax, 'off'); end;

if nargout==0
    varargout = {};
else
    varargout = {hb, he, hpt, hpl, hpb};
end

end


%colorspec2rgb
%   Convert a color string to an RGB value.
%          b     blue
%          g     green
%          r     red
%          c     cyan
%          m     magenta
%          y     yellow
%          k     black
%          w     white
function color = colorspec2rgb(color)

% Define the lookup table
rgb = [1 0 0; 0 1 0; 0 0 1; 1 1 1; 0 1 1; 1 0 1; 1 1 0; 0 0 0];
colspec = 'rgbwcmyk';

idx = find(colspec==color(1));
if isempty(idx)
    error('colorstr2rgb:InvalidColorString', ...
        'Unknown color string: %s.', color(1));
end

if idx~=3 || length(color)==1,
    color = rgb(idx, :);
elseif length(color)>2,
    if strcmpi(color(1:3), 'bla')
        color = [0 0 0];
    elseif strcmpi(color(1:3), 'blu')
        color = [0 0 1];
    else
        error('colorstr2rgb:UnknownColorString', 'Unknown color string.');
    end
end

end


%bar2grouped
%   Split vector X to position all bars in a group into appropriate places.
function [X, Y, width] = bar2grouped(X, Y, width, group_width)

% Parameters
if nargin<4
    group_width = 0.75;
end

nElePerGroup = size(Y, 2);

if nElePerGroup==1
    % No need to do anything to X as the groups only contain one element
    return;
end

if size(X, 1)==1
    % Transpose X if necessary
    X = X';
end
if ~ismatrix(X) || size(X, 2)>1
    error('X must be a column vector.')
end

% Compute the offset for each bar, such that they are centred correctly
dX = width / nElePerGroup * ((0:nElePerGroup-1) - (nElePerGroup-1)/2);
% Apply the offset to each bar in X
X = bsxfun(@plus, X, dX);
% Reduce width of bars so they only take up group_width as much space, and
% divide what there is evenly between the bars per group
width = width * group_width / nElePerGroup;

end


%plot_p_values_single
%   Plot stars above bars to indicate which are statistically significant.
%   Can be used with bars in either horizontal or vertical direction.
function h = plot_p_values_single(ax, X, Y, E, P, orientation, baseval, ...
    p_threshold, offset, show_ns, show_gt, fixed_text_orientation, ...
    star_icon, use_latex, text_args)

if isempty(E)
    E = zeros(size(Y));
end

% Validate inputs
assert(numel(X)==numel(Y), 'Number of datapoints mismatch {X,Y}.');
assert(numel(X)==numel(E), 'Number of datapoints mismatch {X,E}.');
assert(numel(X)==numel(P), 'Number of datapoints mismatch {X,P}.');
assert(all(E(:) >= 0), 'Error must be a non-negative value.');
assert(offset > 0, 'Offset must be a positive value.');

% Deal with axes
if ~isempty(ax)
    % We can't pass an axes argument to the line function because it only
    % became available in R2016a, so instead we change axes if necessary.
    % Make a cleanup object to revert focus back to previous axes
    prev_ax = gca();
    finishup = onCleanup(@() axes(prev_ax));
    % Change focus to the axes we want to work on
    axes(ax);
end

% Loop over every bar
h = nan(size(X));
for i=1:numel(X)
    % Check how many stars to put in the text
    num_stars = sum(P(i) <= p_threshold);
    str = char(repmat(star_icon, 1, num_stars));
    % Check whether to include a > sign too
    if show_gt && all(P(i) < p_threshold) && numel(str) > 1
        str = ['>' str(1:end-1)];
    end
    % Wrap for LaTeX, if its on and we have something to wrap
    if ~strcmp(use_latex, 'off') && ~isempty(str)
        str = ['$' str '$'];
    end
    % Check whether to write n.s. above non-significant bars
    if show_ns && num_stars == 0 && ~isnan(P(i))
        str = 'n.s.';
    end
    % Work out where to put the text
    x = X(i);
    y = Y(i);
    if strncmpi(orientation, 'h', 1)
        % Swap x and y
        tmp = x;
        x = y;
        y = tmp;
        if fixed_text_orientation
            HorizontalAlignment = 'left';
            rotation = 0;
        else
            HorizontalAlignment = 'center';
            rotation = 270;
        end
        % Add on the offset to the x co-ordinate
        if x >= baseval
            x = x + E(i) + offset;
        else
            x = x - E(i) - offset;
        end
    else
        HorizontalAlignment = 'center';
        rotation = 0;
        if y >= baseval
            y = y + E(i) + offset;
        else
            y = y - E(i) - offset;
        end
    end
    % Add the text for the stars
    h(i) = text(x, y, str, ...
        'HorizontalAlignment', HorizontalAlignment, ...
        'VerticalAlignment', 'middle', ...
        'Rotation', rotation, ...
        text_args{:});
    % Change the interpreter to LaTeX, if desired
    if strcmp(use_latex, 'all') || ...
            (strcmp(use_latex, 'auto') && num_stars > 0)
        set(h(i), 'Interpreter', 'latex');
    end
end

end


%plot_p_values_pairs
%   Plot lines and stars to indicate pairwise comparisons and whether they
%   are significant. Only works for error bars in the Y-direction.
function [ht, hl, hbl] = plot_p_values_pairs(ax, X, Y, E, P, orientation, ...
    p_threshold, offset, source_offset, star_offset, show_ns, show_gt, ...
    max_dx_single, max_dx_full, pad_lines, fixed_text_orientation, ...
    star_icon, use_latex, line_args, pad_args, text_args)

if isempty(E)
    E = zeros(size(Y));
end

% Validate inputs
N = numel(X);
assert(numel(Y)==N, 'Number of datapoints mismatch {X,Y}.');
assert(numel(E)==N, 'Number of datapoints mismatch {X,E}.');
assert(numel(P)==N^2, 'Number of datapoints mismatch {X,P}.');

% Deal with axes
if ~isempty(ax)
    % We can't pass an axes argument to the line function because it only
    % became available in R2016a, so instead we change axes if necessary.
    % Make a cleanup object to revert focus back to previous axes
    prev_ax = gca();
    finishup = onCleanup(@() axes(prev_ax));
    % Change focus to the axes we want to work on
    axes(ax);
end

% Turn into vectors
X = X(:);
Y = Y(:);
E = E(:);
P = reshape(P, N, N);
assert(all(all(P==P' | isnan(P))), 'P must be symmetric between pairs');

% Sort by bar location
[X, IX] = sort(X);
Y = Y(IX);
E = E(IX);
P = P(IX, IX);

% Ensure P is symmetric
P = max(P, P');
% Remove lower triangle
P(logical(tril(ones(size(P))))) = NaN;

% Find the max of each pair of bars
pair_max_y = max(repmat(Y + E, 1, N), repmat(Y' + E', N, 1));
% Find the distance between the bars
pair_distance = abs(repmat(X, 1, N) - repmat(X', N, 1));
% Remove pairs which are not measured
li = isnan(P);
pair_max_y(li) = NaN;
pair_distance(li) = NaN;

% Sort by maximum value, smallest first
[~, I1] = sort(pair_max_y(:));
pair_distance_tmp = pair_distance(I1);
% Sort by pair distance
[~, I2] = sort(pair_distance_tmp);
% Combine the two mappings into a single indexing step
IS = I1(I2);
% Now we have primary sort by pair_distance and secondary sort by max value
[ISi, ISj] = ind2sub(size(pair_distance), IS);

% For each bar, check how many lines there will be
num_comp_per_bar = sum(~isnan(max(P, P')), 2);
dX_list = nan(size(P));
for i=1:numel(X)
    dX_list(1:num_comp_per_bar(i), i) = ...
        (0:(num_comp_per_bar(i)-1)) - (num_comp_per_bar(i)-1) / 2;
end
dX_each = min(max_dx_single, max_dx_full / max(num_comp_per_bar));
dX_list = dX_list * dX_each;

% Minimum value for lines over each bar
YEO = Y + E;
current_height = repmat(YEO(:)', N, 1);

% Loop over every pair with a measurement
num_comparisons = sum(~isnan(P(:)));
hl = nan([size(P), 2]);
ht = nan(size(P));
hbl = nan(size(P));
coords = nan(4, 2, num_comparisons);
for iPair=1:num_comparisons
    % Get index of left and right pairs
    i = min(ISi(iPair), ISj(iPair));
    j = max(ISi(iPair), ISj(iPair));
    % Check we're not failing terribly
    if isnan(P(i,j))
        error('This shouldnt be NaN!');
    end
    % Check which bar origin point we're up to
    il = find(~isnan(dX_list(:, i)), 1, 'last');
    jl = find(~isnan(dX_list(:, j)), 1, 'first');
    % Offset the X value to get the non-intersecting origin point
    xi = X(i) + dX_list(il, i);
    xj = X(j) + dX_list(jl, j);
    % Clear these origin points so they aren't reused
    dX_list(il, i) = NaN;
    dX_list(jl, j) = NaN;
    xx = [xi, xi, xj, xj];
    % Work out how high the line must be
    if dX_each==0
        yi = current_height(il, i) + source_offset;
        yj = current_height(jl, j) + source_offset;
    else
        yi = YEO(i) + source_offset;
        yj = YEO(j) + source_offset;
    end
    % It must be higher than all intermediate lines; check which these are
    if dX_each==0
        intermediate_index = (1 + N*(i-1)) : ( N*j );
    else
        intermediate_index = (il + N*(i-1)) : (jl + N*(j-1));
    end
    % Also offset so we are higher than these lines
    y_ = max(current_height(intermediate_index)) + offset;
    yy = [yi, y_, y_, yj];
    % Update intermediates so we know the new hight above them
    current_height(intermediate_index) = y_;
    % Save the co-ordinates to plot later
    coords(:, 1, iPair) = xx;
    coords(:, 2, iPair) = yy;
end

% Plot backing, text and comparison lines
for iPair=num_comparisons:-1:1
    % Get index of left and right pairs
    i = min(ISi(iPair), ISj(iPair));
    j = max(ISi(iPair), ISj(iPair));
    % Get co-ordinates back again
    if strncmpi(orientation, 'h', 1)
        xx = coords(:, 2, iPair);
        yy = coords(:, 1, iPair);
        x_offset = star_offset;
        y_offset = 0;
        if fixed_text_orientation
            HorizontalAlignment = 'left';
            rotation = 0;
        else
            HorizontalAlignment = 'center';
            rotation = 270;
        end
    else
        xx = coords(:, 1, iPair);
        yy = coords(:, 2, iPair);
        x_offset = 0;
        y_offset = star_offset;
        HorizontalAlignment = 'center';
        rotation = 0;
    end
    % Draw the backing line
    if pad_lines
        hbl(i, j) = line(xx([2 3]), yy([2 3]), pad_args{:});
    end
    % Check how many stars to put in the text
    num_stars = sum(P(i, j) <= p_threshold);
    str = char(repmat(star_icon, 1, num_stars));
    % Check whether to include a > sign too
    if show_gt && all(P(i, j) < p_threshold) && numel(str) > 1
        str = ['>' str(1:end-1)];
    end
    % Wrap for LaTeX, if its on and we have something to wrap
    if ~strcmp(use_latex, 'off') && ~isempty(str)
        str = ['$' str '$'];
    end
    % Check whether to write n.s. above non-significant comparisons
    if show_ns && num_stars == 0
        str = 'n.s.';
    end
    % Add the text for the stars, slightly above the middle of the line
    ht(i, j) = text(...
        mean(xx([2 3])) + x_offset, mean(yy([2 3])) + y_offset, ...
        str, ...
        'HorizontalAlignment', HorizontalAlignment, ...
        'VerticalAlignment', 'middle', ...
        'Rotation', rotation, ...
        text_args{:});
    % Change the interpreter to LaTeX, if desired
    if strcmp(use_latex, 'all') || ...
            (strcmp(use_latex, 'auto') && num_stars > 0)
        set(ht(i, j), 'Interpreter', 'latex');
    end
    % Draw the main lines
    hl(i, j, 1) = line(xx, yy, line_args{:});
end

% Plot the horizontal lines again, on top of everything else (specifically
% so they are on top of the text)
for iPair=num_comparisons:-1:1
    % Get index of left and right pairs
    i = min(ISi(iPair), ISj(iPair));
    j = max(ISi(iPair), ISj(iPair));
    % Get co-ordinates back again
    if strncmpi(orientation, 'h', 1)
        xx = coords(:, 2, iPair);
        yy = coords(:, 1, iPair);
    else
        xx = coords(:, 1, iPair);
        yy = coords(:, 2, iPair);
    end
    % Draw the horizontal lines again on top of everything else
    hl(i, j, 2) = line(xx([2 3]), yy([2 3]), line_args{:});
end

end