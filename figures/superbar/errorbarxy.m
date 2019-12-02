function [varargout]=errorbarxy(varargin)

linewidth = 6.0;
%   ERRORBARXY is a function to generate errorbars on both x and y axes 
%   with specified errors modified from codes written by Nils Sj?berg 
%   (http://www.mathworks.com/matlabcentral/fileexchange/5444-xyerrorbar)
%  
%   errorbarxy(x, y, lerrx, uerrx, lerry, uerry) plots the data with errorbars on both 
%   x and y axes with error bars [x-lerrx, x+uerrx] and [y-lerry, y+uerry]. If there is
%   no error on one axis, set corresponding lower and upper bounds to [].
%
%   errorbarxy(x, y, errx, erry) plots the data with errorbars on both x and
%   y axes with error bars [x-errx, x+errx] and [y-erry, y+erry]. If there 
%   is no error on one axis, set corresponding errors to [].
%   
%   errorbarxy(..., S) plots data as well as errorbars using specified
%   formatting string. S is a cell array of 3 element, {sData, cEBx, cEBy},
%   where sData specifies the format of main plot, cEBx specifies the 
%   color of errorbars along x axis and cEBy specifies the color of 
%   errorbars along y axis. The formatting string for the main plot made 
%   from one element from any or all the following 3 columns, while the 
%   other two strings made only from the first colume (color):
%            b     blue          .     point              -     solid
%            g     green         o     circle             :     dotted
%            r     red           x     x-mark             -.    dashdot 
%            c     cyan          +     plus               --    dashed   
%            m     magenta       *     star             (none)  no line
%            y     yellow        s     square
%            k     black         d     diamond
%            w     white         v     triangle (down)
%                                ^     triangle (up)
%                                <     triangle (left)
%                                >     triangle (right)
%                                p     pentagram
%                                h     hexagram
% 
%   
%   errorbarxy(AX,...) plots into AX instead of GCA.
%   
%   H = errorbar(...) returns a vector of errorbarseries handles in H,
%   within which the first element is the handle to the main data plot and
%   the remaining elements are handles to the rest errorbars.
%   H is organized as follows:
%   H.hMain is the handle of the main plot
%   H.hErrorbar is a Nx6 matrix containing handles for all error bar lines,
%               where N is the number of samples. For each sample, 6
%               errorbar handles are saved in such an order:
%               [Horizontal bar, H bar left cap, H bar right cap, 
%                Vertical bar, V bar lower cap, V bar upper cap]
%
%   For example
%       x = 1:10;
%       xe = 0.5*ones(size(x));
%       y = sin(x);
%       ye = std(y)*ones(size(x));
%       H=errorbarxy(x,y,xe,ye,{'ko-', 'b', 'r'});
%    draws symmetric error bars on both x and y axes.
%
%   NOTE: errorbars are excluded from legend display. If you need to
%   include errorbars in legend display, do the followings:
%       H=errorbarxy(...);
%       arrayfun(@(d) set(get(get(d,'Annotation'),'LegendInformation'),...
%       'IconDisplayStyle','on'), H(2:end)); % include errorbars
%       hEB=hggroup;
%       set(H(2:end),'Parent',hEB);
%       set(get(get(hEB,'Annotation'),'LegendInformation'),...
%       'IconDisplayStyle','on'); % include errorbars in legend as a group.
%       legend('Main plot', 'Error bars');
%
%   Developed under Matlab version 7.10.0.499 (R2010a)
%   Created by Qi An
%   anqi2000@gmail.com
%   QA 2/7/2013 initial skeleton
%   QA 2/12/2013    Added support to plot on specified axes; Added support
%                   to specify color of plots and errorbars; Output a
%                   vector of errbar series handles; Fixed a couple of 
%                   minor bugs. 
%   QA 2/13/2013    Excluded errorbars from legend display.
%   QA 8/19/2013    Fixed a bug in errorbar cap display.
%   QA 9/24/2013    Fixed a bug in figure handle output.
%   QA 1/16/2014    Reorganize the output handle structure.
%   QA 2/4/2014     Allow customization on main plot using formatting
%                   string. 
%   QA 4/28/2014    Check for formatting string. Fixed a bug when
%                   specifying axis to plot. 
%   QA 11/23/2015   Fixed a bug when some of
%                   errx/erry/lerrx/uerrx/lerry/uerry are not specified the
%                   code will error out. 
%   QA 11/23/2015   The code will "remember" if the axis is set to be "HOLD ON".
%% handle inputs
if ishandle(varargin{1}) % first argument is a handle
    if strcmpi(get(varargin{1}, 'type'), 'axes') % the handle is for an axes
        axes(varargin{1}); % set the handle to be current
    
        varargin(1)=[];
    end
end
if length(varargin)<4
    error('Insufficient number of inputs');
    return;
end
%% assign values
x=varargin{1};
y=varargin{2};
if length(x)~=length(y)
    error('x and y must have the same number of elements!')
    return
end
if iscell(varargin{end}) & length(varargin{end})==3 % search for formatting strings
    color=varargin{end};
    varargin(end)=[]; % remove formatting string from optional input structure
else
    color={'b', 'r', 'r'};
end
if length(varargin)==4 % using errorbarxy(x, y, errx, erry)
    errx=varargin{3};
    erry=varargin{4};
    if ~isempty(errx)
        lx=x-errx;
        ux=x+errx;
    else
        lx=[];
        ux=[];
    end
    if ~isempty(erry)
        ly=y-erry;
        uy=y+erry;
    else
        ly=[];
        uy=[];
    end
    
elseif length(varargin)==6 % using errorbarxy(x, y, lerrx, uerrx, lerry, uerry)
    lx=x-varargin{3};
    ux=x+varargin{4};
    ly=y-varargin{5};
    uy=y+varargin{6};
    if ~isempty(lx)
        errx=(ux-lx)/2;
    else
        errx=[];
    end
    if ~isempty(ly)
        erry=(uy-ly)/2;
    else
        erry=[];
    end
    
else
    error('Wrong number of inputs!');
end
%% plot data and errorbars
flagHold=ishold; % record the current status of "HOLD"
% h=scatter(x,y, color{1}); % main plot
h=scatter(x, y, 1, 'k', 'filled'); % main plot
allh=nan(length(x), 6); % all errorbar handles
for k=1:length(x)
    if ~isempty(lx) & ~isempty(ly) % both errors are specified
        l1=line([lx(k) ux(k)],[y(k) y(k)], 'LineWidth', linewidth);
        hold on;
        l2=line([lx(k) lx(k)],[y(k)-0.1*erry(k) y(k)+0.1*erry(k)], 'LineWidth', linewidth);
        l3=line([ux(k) ux(k)],[y(k)-0.1*erry(k) y(k)+0.1*erry(k)], 'LineWidth', linewidth);
        l4=line([x(k) x(k)],[ly(k) uy(k)], 'LineWidth', linewidth);
        l5=line([x(k)-0.1*errx(k) x(k)+0.1*errx(k)],[ly(k) ly(k)], 'LineWidth', linewidth);
        l6=line([x(k)-0.1*errx(k) x(k)+0.1*errx(k)],[uy(k) uy(k)], 'LineWidth', linewidth);
        allh(k, :)=[l1, l2, l3, l4, l5, l6];
    elseif isempty(lx) & ~isempty(ly) % x errors are not specified
        l4=line([x(k) x(k)],[ly(k) uy(k)]);
        hold on;
        errx=nanmean(abs(diff(x)));
        l5=line([x(k)-0.1*errx x(k)+0.1*errx],[ly(k) ly(k)], 'LineWidth', linewidth);
        l6=line([x(k)-0.1*errx x(k)+0.1*errx],[uy(k) uy(k)], 'LineWidth', linewidth);
        allh(k, 4:6)=[l4, l5, l6];
    elseif ~isempty(lx) & isempty(ly) % y errors are not specified
        l1=line([lx(k) ux(k)],[y(k) y(k)]);
        hold on;
        erry=nanmean(abs(diff(y)));
        l2=line([lx(k) lx(k)],[y(k)-0.1*erry y(k)+0.1*erry], 'LineWidth', linewidth);
        l3=line([ux(k) ux(k)],[y(k)-0.1*erry y(k)+0.1*erry], 'LineWidth', linewidth);
        allh(k, 1:3)=[l1, l2, l3];
    else % both errors are not specified
    end
    if exist('l1', 'var')
        h1=[l1, l2, l3]; % all handles
        set(h1, 'color', color{2});
    end
    if exist('l4', 'var')
        h1=[l4, l5, l6]; % all handles
        set(h1, 'color', color{3});
    end
end
arrayfun(@(d) set(get(get(d,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off'), allh(~isnan(allh))); % exclude errorbars from legend
out.hMain=h;
out.hErrorbar=allh;
hold off;
if flagHold % set the axis to the original status, e.g hold on or off
    hold on;
end
%% handle outputs
if nargout>0
    varargout{1}=out;
end
