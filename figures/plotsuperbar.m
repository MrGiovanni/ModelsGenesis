%% Source: https://github.com/scottclowe/superbar

tab_space = 0.03;
LineWidth = 5;

%% 2D Genesis Chest CT T4
font_size = 45;
target_name = {'NCC'; ...
               'NCS'; ...
               'DXC'; ...
               'IUC'};
hf = figure('Position', [0 0 1200 700]);

Y = [ 96.03 97.64 97.45;
      70.48 72.39 72.20;
      69.15 75.55 74.99;
      92.78 95.51 94.50]; 
E = [ 0.86 0.96 0.61;
      1.07 0.77 0.67;
      0.77 0.48 0.36;
      3.71 1.77 2.08];

C = [
    92/255 102/255 112/255
    92/255 102/255 112/255
    92/255 102/255 112/255
    92/255 102/255 112/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    ];
C = reshape(C, [length(target_name) 3 3]);

P = nan(numel(Y), numel(Y));

P(1*length(target_name)-3,2*length(target_name)-3) = 0.0005;
P(1*length(target_name)-3,3*length(target_name)-3) = 0.0002;
P(2*length(target_name)-3,3*length(target_name)-3) = 0.3048;

P(1*length(target_name)-2,2*length(target_name)-2) = 0.0001;
P(1*length(target_name)-2,3*length(target_name)-2) = 0.0010;
P(2*length(target_name)-2,3*length(target_name)-2) = 0.3042;

P(1*length(target_name)-1,2*length(target_name)-1) = 0.0000001;
P(1*length(target_name)-1,3*length(target_name)-1) = 0.00000000001;
P(2*length(target_name)-1,3*length(target_name)-1) = 0.0121;

P(1*length(target_name)-0,2*length(target_name)-0) = 0.0248;
P(1*length(target_name)-0,3*length(target_name)-0) = 0.1093;
P(2*length(target_name)-0,3*length(target_name)-0) = 0.1262;

% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);

superbar(Y, 'E', E, 'P', P, 'BarFaceColor', C, 'Orientation', 'v', ...
    'ErrorbarStyle', 'I', 'ErrorbarLineWidth', 3, 'PLineOffset', 3, 'PStarFontSize', font_size*0.6, ...
    'PLineColor', [0 0 0], 'BarEdgeColor', 'k', 'PStarLatex', 'on', ...
    'PStarLatex', 'off', 'PStarIcon', char(10033), ...
    'BarWidth', 0.95, 'PLineWidth', 3, 'BarLineWidth', 3);

xlim([0.5 length(target_name)+0.5]);
ylim([60 108]);
ax = gca;
ax.YGrid = 'on';
ax.GridLineStyle = '-';
ax.LineWidth = 3;
ax.GridColor = 'k';
ax.GridAlpha = 0.3; % maximum line opacity
ht = text(1.75, 96, ...
          {'{\color[rgb]{0.8157,0.2078,0.1882} I } Genesis', ...
           '{\color[rgb]{1,0.7765,0.1529} I } ImageNet', ...
           '{\color[rgb]{0.3608,0.4000,0.4392} I } Scratch', ...
           'n.s.: No Significance', ...
           '    *: p<0.05       **: p<0.01', ...
           '***: p<0.001 ****: p<0.0001'}, ...
          'EdgeColor', 'k', 'LineWidth', LineWidth*0.6, 'FontSize', font_size*0.75, ...
          'BackgroundColor', 'w', 'FontName', 'times');
set(gca, 'xtick', 1:length(target_name), 'xticklabel',target_name, 'FontSize', font_size);
set(gca, 'YTick', 60:10:100);
ylabel('Performance'); xlabel('Target Tasks'); title('Genesis Chest CT 2D');
set(gca,'fontname','times', 'LooseInset',get(gca,'TightInset'))  % Set it to times
print(gcf,'Fig_2D_Chest_CT_T4_legend.png','-dpng','-r100');



%% 2D Genesis Chest X-ray T4
target_name = {'NCC'; ...
               'NCS'; ...
               'DXC'; ...
               'IUC'; ...
               };
hf = figure('Position', [0 0 1200 700]);

Y = [ 96.03 97.64 93.52;
      70.48 72.39 72.88;
      69.15 75.55 75.48;
      92.78 95.52 94.99]; 
E = [ 0.86 0.96 0.13;
      1.07 0.77 0.79;
      0.77 0.48 0.13;
      3.71 1.77 0.93];

C = [
    92/255 102/255 112/255
    92/255 102/255 112/255
    92/255 102/255 112/255
    92/255 102/255 112/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    255/255 198/255 39/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    208/255 53/255 48/255
    ];
C = reshape(C, [length(target_name) 3 3]);

P = nan(numel(Y), numel(Y));

P(1*length(target_name)-3,2*length(target_name)-3) = 0.0005;
P(1*length(target_name)-3,3*length(target_name)-3) = 0.0096;
P(2*length(target_name)-3,3*length(target_name)-3) = 0.0001;

P(1*length(target_name)-2,2*length(target_name)-2) = 0.0001;
P(1*length(target_name)-2,3*length(target_name)-2) = 0.0022;
P(2*length(target_name)-2,3*length(target_name)-2) = 0.1006;

P(1*length(target_name)-1,2*length(target_name)-1) = 0.0000001;
P(1*length(target_name)-1,3*length(target_name)-1) = 0.00000005;
P(2*length(target_name)-1,3*length(target_name)-1) = 0.3736;

P(1*length(target_name)-0,2*length(target_name)-0) = 0.0248;
P(1*length(target_name)-0,3*length(target_name)-0) = 0.0423;
P(2*length(target_name)-0,3*length(target_name)-0) = 0.2078;

% Make P symmetric, by copying the upper triangle onto the lower triangle
PT = P';
lidx = tril(true(size(P)), -1);
P(lidx) = PT(lidx);

superbar(Y, 'E', E, 'P', P, 'BarFaceColor', C, 'Orientation', 'v', ...
    'ErrorbarStyle', 'I', 'ErrorbarLineWidth', 3, 'PLineOffset', 3, 'PStarFontSize', font_size*0.6, ...
    'PLineColor', [0 0 0], 'BarEdgeColor', 'k', 'PStarLatex', 'on', ...
    'PStarLatex', 'off', 'PStarIcon', char(10033), ...
    'BarWidth', 0.95, 'PLineWidth', 3, 'BarLineWidth', 3);

xlim([0.5 length(target_name)+0.5]);
ylim([60 108]);
ax = gca;
ax.YGrid = 'on';
ax.GridLineStyle = '-';
ax.LineWidth = 3;
ax.GridColor = 'k';
ax.GridAlpha = 0.3; % maximum line opacity
ht = text(1.75, 96, ...
          {'{\color[rgb]{0.8157,0.2078,0.1882} I } Genesis', ...
           '{\color[rgb]{1,0.7765,0.1529} I } ImageNet', ...
           '{\color[rgb]{0.3608,0.4000,0.4392} I } Scratch', ...
           'n.s.: No Significance', ...
           '    *: p<0.05       **: p<0.01', ...
           '***: p<0.001 ****: p<0.0001'}, ...
          'EdgeColor', 'k', 'LineWidth', LineWidth*0.6, 'FontSize', font_size*0.75, ...
          'BackgroundColor', 'w', 'FontName', 'times');
set(gca, 'xtick', 1:length(target_name), 'xticklabel',target_name, 'FontSize', font_size);
set(gca, 'YTick', 60:10:100);
ylabel('Performance'); xlabel('Target Tasks'); title('Genesis Chest X-ray (2D)');
set(gca,'fontname','times', 'LooseInset',get(gca,'TightInset'))  % Set it to times
print(gcf,'Fig_2D_Chest_Xray_T4_legend.png','-dpng','-r100');


