function matrix2latex_Stefan(matrix, filename, varargin)

% function: matrix2latex(...)
% Author:   M. Koehler
% Contact:  koehler@in.tum.de
% Version:  1.1
% Date:     May 09, 2004

% modified by Stefan Nagel June 2007

% This software is published under the GNU GPL, by the free software
% foundation. For further reading see: http://www.gnu.org/licenses/licenses.html#GPL

% Usage:
% matrix2late(matrix, filename, varargs)
% where
%   - -9999 is a code for an empty cell
%   - matrix is a 2 dimensional numerical or cell array
%   - filename is a valid filename, in which the resulting latex code will
%     be stored
%   - varargs is one ore more of the following (denominator, value) combinations
%      + 'rowLabels', array -> Can be used to label the rows of the
%      resulting latex table
%      + 'columnLabels', array -> Can be used to label the columns of the
%      resulting latex table
%      + 'alignment', 'value' -> Can be used to specify the alginment of
%      the table within the latex document. Valid arguments are: 'l', 'c',
%      and 'r' for left, center, and right, respectively
%      + 'format', 'value' -> Can be used to format the input data. 'value'
%      has to be a valid format string, similar to the ones used in
%      fprintf('format', value);
%      + 'size', 'value' -> One of latex' recognized font-sizes, e.g. tiny,
%      HUGE, Large, large, LARGE, etc.
%      + 'specform', array -> 'p' for parentheses, 'b' for brackets, ' '
%      otherwise
%      + 'topwidth', array -> additional header line w/ merged cells
%      + 'topcollabels', array -> labels for additional header line
%      + 'topcollines', array -> '_' for underlined cells, ' ' otherwise
%
% Example input:
%   matrix = [1.5 1.764; 3.523 0.2];
%   rowLabels = {'row 1', 'row 2'};
%   columnLabels = {'col 1', 'col 2'};
%   matrix2latex(matrix, 'out.tex', 'rowLabels', rowLabels, 'columnLabels', ...
%         columnLabels, 'alignment', 'c', 'format', '%-6.2f', 'size', 'tiny');
%
% The resulting latex file can be included into any latex document by:
% /input{out.tex}
%



    [height width] = size(matrix);
    
    rowLabels = [];
    colLabels = [];
    alignment = 'l';
    format = [];
    textsize = [];
    topwidth = [];
    topcolLabels = [];
    topcolLines = [];
    specform = repmat(' ', height, width); 
    
    if (rem(nargin,2) == 1 || nargin < 2)
        error('matrix2latex: ', 'Incorrect number of arguments to %s.', mfilename);
    end

    okargs = {'rowlabels', 'columnlabels', 'alignment', 'format', 'size', ...
        'specform', 'topwidth', 'topcollabels', 'topcollines'};
        
    for j=1:2:(nargin-2)
        pname = varargin{j};
        pval = varargin{j+1};
        k = strmatch(lower(pname), okargs);
        if isempty(k)
            error('matrix2latex: ', 'Unknown parameter name: %s.', pname);
        elseif length(k)>1
            error('matrix2latex: ', 'Ambiguous parameter name: %s.', pname);
        else
            switch(k)
                case 1  % rowlabels
                    rowLabels = pval;
                    if isnumeric(rowLabels)
                        rowLabels = cellstr(num2str(rowLabels(:)));
                    end
                case 2  % column labels
                    colLabels = pval;
                    if isnumeric(colLabels)
                        colLabels = cellstr(num2str(colLabels(:)));
                    end
                case 3  % alignment
                    alignment = lower(pval);
                    if alignment == 'right'
                        alignment = 'r';
                    end
                    if alignment == 'left'
                        alignment = 'l';
                    end
                    if alignment == 'center'
                        alignment = 'c';
                    end
                    if alignment ~= 'l' && alignment ~= 'c' && alignment ~= 'r'
                        alignment = 'l';
                        warning('matrix2latex: ', 'Unkown alignment. (Set it to \''left\''.)');
                    end
                case 4  % format
                    format = lower(pval);
                case 5  % format
                    textsize = pval;
                case 6  % brackets or parentheses
                    specform = pval; 
                case 7  % top header
                    topwidth = pval; 
                case 8  % top header
                    topcolLabels = pval;
                    if isnumeric(topcolLabels)
                        topcolLabels = cellstr(num2str(topcolLabels(:)));
                    end
                case 9  % top header
                    topcolLines = pval;                 
            end
        end
    end

    fid = fopen(filename, 'wt');
           
    if isnumeric(matrix)
        matrix = num2cell(matrix);
        
        for h=1:height
            for w=1:width
                if(~isempty(format))
                    if matrix{h, w} == -9999
                        matrix{h, w} = ' '; 
                    elseif specform(h,w) == 'p' 
                       matrix{h, w} = ['(',num2str(matrix{h, w}, format),')'];
                    elseif specform(h,w) == 'b' 
                       matrix{h, w} = ['[',num2str(matrix{h, w}, format),']'];
                    else   
                       matrix{h, w} = num2str(matrix{h, w}, format);
                    end
                else
                    if matrix{h, w} == -9999
                        matrix{h, w} = ' '; 
                    elseif specform(h,w) == 'p' 
                        matrix{h, w} = ['(',num2str(matrix{h, w}),')'];
                    elseif specform(h,w) == 'b' 
                        matrix{h, w} = ['[',num2str(matrix{h, w}),']'];
                    else 
                        matrix{h, w} = num2str(matrix{h, w});
                    end
                end
            end
        end
    end
    
    if(~isempty(textsize))
        fprintf(fid, '\\begin{%s}', textsize);
    end

    fprintf(fid, '\\begin{tabular}{' );

    if(~isempty(rowLabels))
        fprintf(fid, 'l');
    end
    for i=1:width
        fprintf(fid, '%c', alignment);  %Note: %c = single character format 
    end
    fprintf(fid, '}\r\n');
    
    fprintf(fid, '\\hline\\hline\r\n');
   
    if(~isempty(topcolLabels))
        if(~isempty(rowLabels))
            fprintf(fid, '& ');
        end
        count = 1; 
        subcount = 1; 
        for tw = 1:width
           if topwidth(count) == 1
               if tw == width
                  fprintf(fid, '%s ', topcolLabels{count});
               else
                  fprintf(fid, '%s& ', topcolLabels{count}); 
               end
               count = count+1;
           else
               if subcount == topwidth(count)   %& \multicolumn{5}{c}{head1} &  & head2 \\ \cline{2-6}\cline{8-8}
                   twcstr = int2str(subcount);
                   fprintf(fid, '\\multicolumn{%s}{c}', twcstr);
                   if tw == width
                      fprintf(fid, '{%s} ', topcolLabels{count});
                   else
                      fprintf(fid, '{%s} & ', topcolLabels{count}); 
                   end
                   subcount = 1; 
                   count = count+1; 
               else
                   subcount = subcount+1;
               end
           end
        end
        fprintf(fid, '\\\\ ');
        count = 1; 
        subcount = 1; 
        if(~isempty(rowLabels))
            shift = 1;
        else 
            shift = 0;
        end
        for tw = 1:width
           if topwidth(count) == 1
               if topcolLines(count) == '_'
                      startstr = int2str(tw+shift); 
                      endstr = int2str(tw+shift); 
                      fprintf(fid, '\\cline{%s',startstr);
                      fprintf(fid, '-%s}', endstr);
               end  
               count = count+1;
           else
               if subcount == topwidth(count)
                   startstr = int2str(tw-subcount+1+shift);
                   endstr = int2str(tw+shift); 
                   if topcolLines(count) == '_'
                      fprintf(fid, '\\cline{%s',startstr);
                      fprintf(fid, '-%s}', endstr);
                   end
                   subcount = 1;
                   count = count+1;
               else
                   subcount = subcount+1;
               end
           end
        end 
        fprintf(fid, '\r\n'); 
    end
     
         
    if(~isempty(colLabels))
        if(~isempty(rowLabels))
            fprintf(fid, '& ');
        end
        for w=1:width-1
            fprintf(fid, '%s &', colLabels{w});   %Note: %s = string format, \\ results in a backslash 
        end
        fprintf(fid, '%s\\\\\\hline\r\n', colLabels{width});   %Note: \r\n is carriage return and new line
    end                                                        %   \\ are row separators in tex, to 
                                                               %   generate them in Matlab requires four \ 
    for h=1:height
        if(~isempty(rowLabels))
            fprintf(fid, '%s &', rowLabels{h});
        end
        for w=1:width-1
            fprintf(fid, '%s &', matrix{h, w});
        end
        fprintf(fid, '%s\\\\\r\n', matrix{h, width});
    end
    
    fprintf(fid, '\\hline\\hline\r\n');
    fprintf(fid, '\\end{tabular}\r\n');
    
    if(~isempty(textsize))
        fprintf(fid, '\\end{%s}', textsize);
    end

    fclose(fid);
